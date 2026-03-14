import "server-only";

import fs from "node:fs/promises";
import path from "node:path";
import Papa from "papaparse";

import type {
  AuditEvent,
  AuditTrade,
  Guardrail,
  MarketCandle,
  MarketTimeframes,
  MarketZone,
  MetricTile,
  ReasoningTree,
  SignalCandidate,
  TerminalMode,
  TerminalSnapshot,
} from "@/lib/terminal-types";

const REPO_ROOT = process.env.QUANT_SMC_ROOT
  ? path.resolve(process.env.QUANT_SMC_ROOT)
  : path.resolve(process.cwd(), "..");
const BACKEND_API_URL = process.env.QUANT_TERMINAL_API_URL ?? "http://127.0.0.1:8100/snapshot";
const BACKTEST_DIR_OVERRIDE = process.env.QUANT_TERMINAL_BACKTEST_DIR ?? "";
const FORWARD_DIR_OVERRIDE = process.env.QUANT_TERMINAL_FORWARD_DIR ?? "";
const LIVE_DIR_OVERRIDE = process.env.QUANT_TERMINAL_LIVE_DIR ?? "";

const BACKTEST_DIR_CANDIDATES = ["backtest_outputs", "backtest_output", "artifacts/backtest/latest"] as const;
const FORWARD_DIR_CANDIDATES = ["forward_outputs", "artifacts/forward/latest"] as const;
const LIVE_DIR_CANDIDATES = ["live_outputs", "artifacts/live/latest"] as const;

const BACKTEST_FILE_MARKERS = [
  "summary.json",
  "ledger.csv",
  "trades.csv",
  "execution_log.csv",
  "candles_15m.csv",
  "candles.csv",
  "reasoning.json",
] as const;

const RUNTIME_FILE_MARKERS = [
  "snapshot.json",
  "state.json",
  "events.json",
  "events.csv",
  "closed_trades.csv",
  "candles.csv",
  "bars.csv",
] as const;

type BacktestBundle = {
  root: string | null;
  summary: Record<string, unknown> | null;
  ledger: Record<string, string>[];
  candles: Record<string, string>[];
  reasoning: Record<string, unknown> | null;
  mtimeMs: number;
};

type RuntimeBundle = {
  root: string | null;
  snapshot: Record<string, unknown> | null;
  state: Record<string, unknown> | null;
  events: Record<string, string>[];
  ledger: Record<string, string>[];
  candles: Record<string, string>[];
  mtimeMs: number;
};

function asReasoningTree(value: unknown): ReasoningTree | undefined {
  return value && typeof value === "object" ? (value as ReasoningTree) : undefined;
}

function normalizeMode(value: unknown, fallback: TerminalMode = "auto"): TerminalMode {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (normalized === "backtest" || normalized === "forward" || normalized === "live" || normalized === "auto") {
    return normalized;
  }
  return fallback;
}

async function exists(filePath: string) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function statMtimeMs(filePath: string): Promise<number> {
  try {
    const stat = await fs.stat(filePath);
    return stat.mtimeMs;
  } catch {
    return 0;
  }
}

function resolveRoot(candidate: string): string {
  return path.isAbsolute(candidate) ? candidate : path.join(REPO_ROOT, candidate);
}

function uniqueRoots(roots: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const root of roots) {
    const normalized = resolveRoot(root.trim());
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

async function chooseBundleRoot(candidates: string[], markers: readonly string[]): Promise<string | null> {
  let fallbackRoot: string | null = null;
  let bestRoot: string | null = null;
  let bestScore = -1;
  let bestMtime = -1;

  for (const candidate of candidates) {
    const root = resolveRoot(candidate);
    if (!(await exists(root))) continue;
    fallbackRoot ??= root;

    let score = 0;
    let latest = 0;
    for (const marker of markers) {
      const full = path.join(root, marker);
      if (await exists(full)) {
        score += 1;
        latest = Math.max(latest, await statMtimeMs(full));
      }
    }
    if (score > bestScore || (score === bestScore && latest > bestMtime)) {
      bestScore = score;
      bestMtime = latest;
      bestRoot = root;
    }
  }

  if (bestRoot && bestScore > 0) return bestRoot;
  return fallbackRoot;
}

async function readFirstJson<T>(root: string | null, candidates: readonly string[]): Promise<T | null> {
  if (!root) return null;
  for (const candidate of candidates) {
    const payload = await readJson<T>(path.join(root, candidate));
    if (payload) return payload;
  }
  return null;
}

async function readFirstCsv(root: string | null, candidates: readonly string[]): Promise<Record<string, string>[]> {
  if (!root) return [];
  for (const candidate of candidates) {
    const rows = await readCsv(path.join(root, candidate));
    if (rows.length) return rows;
  }
  return [];
}

function flattenPayloadRow(value: unknown): Record<string, string> {
  if (!value || typeof value !== "object") return {};
  const row = value as Record<string, unknown>;
  const payload = row.payload && typeof row.payload === "object"
    ? (row.payload as Record<string, unknown>)
    : {};
  const merged = { ...payload, ...row };
  const flattened: Record<string, string> = {};
  for (const [key, entry] of Object.entries(merged)) {
    if (entry === undefined || entry === null) continue;
    if (typeof entry === "object") {
      flattened[key] = JSON.stringify(entry);
    } else {
      flattened[key] = String(entry);
    }
  }
  return flattened;
}

function stateClosedTrades(state: Record<string, unknown> | null): Record<string, string>[] {
  if (!state || typeof state.closed_trades !== "object" || state.closed_trades === null) return [];
  return Object.values(state.closed_trades as Record<string, unknown>)
    .filter((trade): trade is Record<string, unknown> => Boolean(trade) && typeof trade === "object")
    .map((trade) => flattenPayloadRow(trade));
}

function runtimeHasActivity(bundle: RuntimeBundle): boolean {
  if (bundle.events.length || bundle.ledger.length || bundle.candles.length) return true;
  if (bundle.snapshot && Object.keys(bundle.snapshot).length) return true;
  if (bundle.state && Object.keys(bundle.state).length) return true;
  return false;
}

function resolveEffectiveMode(
  requestedMode: TerminalMode,
  forwardBundle: RuntimeBundle,
  liveBundle: RuntimeBundle,
): TerminalMode {
  if (requestedMode !== "auto") return requestedMode;
  const forwardActive = runtimeHasActivity(forwardBundle);
  const liveActive = runtimeHasActivity(liveBundle);
  if (liveActive && !forwardActive) return "live";
  if (forwardActive && !liveActive) return "forward";
  if (liveActive && forwardActive) {
    return liveBundle.mtimeMs >= forwardBundle.mtimeMs ? "live" : "forward";
  }
  return "backtest";
}

async function readJson<T>(filePath: string): Promise<T | null> {
  if (!(await exists(filePath))) return null;
  try {
    return JSON.parse(await fs.readFile(filePath, "utf8")) as T;
  } catch {
    return null;
  }
}

async function readCsv(filePath: string): Promise<Record<string, string>[]> {
  if (!(await exists(filePath))) return [];
  const raw = await fs.readFile(filePath, "utf8");
  const parsed = Papa.parse<Record<string, string>>(raw, { header: true, skipEmptyLines: true });
  return parsed.data;
}

async function listLatestModelVersion(modelRoot: string): Promise<string> {
  if (!(await exists(modelRoot))) return "demo-v1";
  const dirents = await fs.readdir(modelRoot, { withFileTypes: true });
  const versions: string[] = [];
  for (const dirent of dirents) {
    if (!dirent.isDirectory()) continue;
    const inner = await fs.readdir(path.join(modelRoot, dirent.name), { withFileTypes: true }).catch(() => []);
    const localVersions = inner.filter((item) => item.isDirectory()).map((item) => item.name);
    versions.push(...localVersions);
  }
  return versions.sort().at(-1) ?? "unavailable";
}

function fmtMoney(value: number) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);
}

function fmtPct(value: number) {
  return `${value.toFixed(1)}%`;
}

function fmtR(value: number) {
  return `${value.toFixed(2)}R`;
}

function num(value: unknown, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function toUnix(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" && Number.isFinite(value)) {
    return value > 10_000_000_000 ? Math.floor(value / 1000) : Math.floor(value);
  }
  const parsed = Date.parse(String(value));
  return Number.isFinite(parsed) ? Math.floor(parsed / 1000) : null;
}

function toneFromGuardrail(status: Guardrail["status"]): MetricTile["tone"] {
  if (status === "pass") return "teal";
  if (status === "warn") return "amber";
  return "rose";
}

function demoCandles(anchor = 100_000, bars = 520): TerminalSnapshot["market"]["candles"] {
  const now = Math.floor(Date.now() / 1000);
  const step = 15 * 60;
  const rows: TerminalSnapshot["market"]["candles"] = [];
  let prev = anchor;
  for (let i = bars - 1; i >= 0; i -= 1) {
    const time = now - i * step;
    const drift = 0.00015 * ((bars - i) / bars);
    const wave = 0.0026 * Math.sin((bars - i) / 11);
    const shock = 0.001 * Math.cos((bars - i) / 7);
    const ret = drift + wave + shock;
    const close = Math.max(1, prev * (1 + ret));
    const open = prev;
    const high = Math.max(open, close) * 1.0014;
    const low = Math.min(open, close) * 0.9986;
    const volume = 120 + Math.abs(ret) * 85_000;
    rows.push({ time, open, high, low, close, volume });
    prev = close;
  }
  return rows;
}

function parseCandlePayload(data: unknown): TerminalSnapshot["market"]["candles"] {
  let rows: Record<string, unknown>[] = [];
  if (Array.isArray(data)) {
    rows = data as Record<string, unknown>[];
  } else if (data && typeof data === "object") {
    const payload = data as Record<string, unknown>;
    if (Array.isArray(payload.candles)) {
      rows = payload.candles as Record<string, unknown>[];
    } else if (Array.isArray(payload.records)) {
      rows = payload.records as Record<string, unknown>[];
    }
  }
  const mapped = rows
    .map((row) => {
      const time = toUnix(row.time ?? row.timestamp ?? row.ts ?? row.dt);
      if (!time) return null;
      const open = num(row.open ?? row.o, NaN);
      const high = num(row.high ?? row.h, NaN);
      const low = num(row.low ?? row.l, NaN);
      const close = num(row.close ?? row.c, NaN);
      const volume = num(row.volume ?? row.v, 0);
      if (![open, high, low, close].every(Number.isFinite)) return null;
      return { time, open, high, low, close, volume };
    })
    .filter((row): row is TerminalSnapshot["market"]["candles"][number] => row !== null)
    .sort((a, b) => a.time - b.time);

  const deduped: TerminalSnapshot["market"]["candles"] = [];
  let prevTime = -1;
  for (const candle of mapped) {
    if (candle.time === prevTime) {
      deduped[deduped.length - 1] = candle;
    } else {
      deduped.push(candle);
      prevTime = candle.time;
    }
  }
  return deduped.slice(-1200);
}

function aggregateCandles(source: MarketCandle[], barsPerCandle: number, limit = 500): MarketCandle[] {
  if (!source.length) return [];
  if (barsPerCandle <= 1) return source.slice(-limit);

  const out: MarketCandle[] = [];
  const offset = source.length % barsPerCandle;
  for (let i = offset; i < source.length; i += barsPerCandle) {
    const group = source.slice(i, i + barsPerCandle);
    if (!group.length) continue;
    const first = group[0];
    const last = group[group.length - 1];
    let high = first.high;
    let low = first.low;
    let volume = 0;
    for (const row of group) {
      high = Math.max(high, row.high);
      low = Math.min(low, row.low);
      volume += row.volume;
    }
    out.push({
      time: last.time,
      open: first.open,
      high,
      low,
      close: last.close,
      volume,
    });
  }

  return (out.length ? out : source).slice(-limit);
}

function buildTimeframes(candles: MarketCandle[]): MarketTimeframes {
  const m15 = candles.length ? candles.slice(-1200) : demoCandles();
  return {
    m15,
    h1: aggregateCandles(m15, 4, 900),
    h6: aggregateCandles(m15, 24, 700),
    h12: aggregateCandles(m15, 48, 520),
  };
}

function parseTimeframesPayload(data: unknown, fallbackCandles: MarketCandle[]): MarketTimeframes | null {
  if (!data || typeof data !== "object") return null;
  const payload = data as Record<string, unknown>;
  const m15 = parseCandlePayload(payload.m15 ?? payload["15m"] ?? payload.candles);
  const base = m15.length ? m15 : fallbackCandles;
  if (!base.length) return null;

  const h1 = parseCandlePayload(payload.h1 ?? payload["1h"]);
  const h6 = parseCandlePayload(payload.h6 ?? payload["6h"]);
  const h12 = parseCandlePayload(payload.h12 ?? payload["12h"]);
  return {
    m15: base.slice(-1200),
    h1: (h1.length ? h1 : aggregateCandles(base, 4, 900)).slice(-900),
    h6: (h6.length ? h6 : aggregateCandles(base, 24, 700)).slice(-700),
    h12: (h12.length ? h12 : aggregateCandles(base, 48, 520)).slice(-520),
  };
}

function deriveZones(candles: MarketCandle[], signals: SignalCandidate[]): MarketZone[] {
  if (candles.length < 40) return [];

  const tail = candles.slice(-260);
  const anchor = tail[tail.length - 1];
  const ranges = tail.slice(-72).map((row) => Math.max(0, row.high - row.low));
  const avgRange = ranges.length ? ranges.reduce((acc, value) => acc + value, 0) / ranges.length : anchor.close * 0.004;
  const zoneRange = Math.max(avgRange * 1.4, anchor.close * 0.0025);
  const start = tail[Math.max(0, tail.length - 160)]?.time ?? tail[0].time;
  const end = anchor.time + 15 * 60 * 28;

  let pivotHigh = tail[0];
  let pivotLow = tail[0];
  for (const row of tail) {
    if (row.high > pivotHigh.high) pivotHigh = row;
    if (row.low < pivotLow.low) pivotLow = row;
  }

  let bullishGap: { start: number; end: number; top: number; bottom: number } | null = null;
  let bearishGap: { start: number; end: number; top: number; bottom: number } | null = null;
  let bullGapSize = 0;
  let bearGapSize = 0;
  for (let i = 2; i < tail.length; i += 1) {
    const prev = tail[i - 2];
    const curr = tail[i];
    if (curr.low > prev.high) {
      const gap = curr.low - prev.high;
      if (gap > bullGapSize) {
        bullGapSize = gap;
        bullishGap = {
          start: tail[i - 1]?.time ?? curr.time,
          end: curr.time + 15 * 60 * 32,
          top: curr.low,
          bottom: prev.high,
        };
      }
    }
    if (curr.high < prev.low) {
      const gap = prev.low - curr.high;
      if (gap > bearGapSize) {
        bearGapSize = gap;
        bearishGap = {
          start: tail[i - 1]?.time ?? curr.time,
          end: curr.time + 15 * 60 * 32,
          top: prev.low,
          bottom: curr.high,
        };
      }
    }
  }

  const leading = signals[0];
  const bullishBias = !leading || leading.side === "long";
  const zones: MarketZone[] = [
    {
      kind: "ob",
      side: "bullish",
      start,
      end,
      top: anchor.close - zoneRange * 0.3,
      bottom: anchor.close - zoneRange * 1.3,
      label: "Bullish OB",
      score: bullishBias ? 0.84 : 0.66,
    },
    {
      kind: "ob",
      side: "bearish",
      start,
      end,
      top: anchor.close + zoneRange * 1.3,
      bottom: anchor.close + zoneRange * 0.3,
      label: "Bearish OB",
      score: bullishBias ? 0.62 : 0.83,
    },
    {
      kind: "liquidity",
      side: "bearish",
      start: Math.max(start, pivotHigh.time - 15 * 60 * 16),
      end: pivotHigh.time + 15 * 60 * 40,
      top: pivotHigh.high + zoneRange * 0.25,
      bottom: pivotHigh.high - zoneRange * 0.25,
      label: "Buy-side Liquidity",
      score: 0.74,
    },
    {
      kind: "liquidity",
      side: "bullish",
      start: Math.max(start, pivotLow.time - 15 * 60 * 16),
      end: pivotLow.time + 15 * 60 * 40,
      top: pivotLow.low + zoneRange * 0.25,
      bottom: pivotLow.low - zoneRange * 0.25,
      label: "Sell-side Liquidity",
      score: 0.73,
    },
  ];

  if (bullishGap) {
    zones.push({
      kind: "fvg",
      side: "bullish",
      start: bullishGap.start,
      end: bullishGap.end,
      top: bullishGap.top,
      bottom: bullishGap.bottom,
      label: "Bullish FVG",
      score: 0.69,
    });
  }
  if (bearishGap) {
    zones.push({
      kind: "fvg",
      side: "bearish",
      start: bearishGap.start,
      end: bearishGap.end,
      top: bearishGap.top,
      bottom: bearishGap.bottom,
      label: "Bearish FVG",
      score: 0.67,
    });
  }
  return zones;
}

function parseZonesPayload(data: unknown, candles: MarketCandle[], signals: SignalCandidate[]): MarketZone[] {
  const base = candles.length ? candles : demoCandles();
  const derived = deriveZones(base, signals);
  if (!Array.isArray(data)) return derived;

  const fallbackStart = base[Math.max(0, base.length - 120)]?.time ?? base[0]?.time ?? Math.floor(Date.now() / 1000) - 15 * 60 * 120;
  const fallbackEnd = base[base.length - 1]?.time
    ? base[base.length - 1].time + 15 * 60 * 28
    : Math.floor(Date.now() / 1000);

  const parsed = data
    .map((row): MarketZone | null => {
      if (!row || typeof row !== "object") return null;
      const input = row as Record<string, unknown>;
      const kindRaw = String(input.kind ?? input.type ?? "ob").toLowerCase();
      const kind: MarketZone["kind"] = kindRaw.includes("liq")
        ? "liquidity"
        : kindRaw.includes("fvg")
          ? "fvg"
          : "ob";
      const sideRaw = String(input.side ?? input.direction ?? "neutral").toLowerCase();
      const side: MarketZone["side"] = sideRaw.includes("bull")
        ? "bullish"
        : sideRaw.includes("bear")
          ? "bearish"
          : "neutral";
      const start = toUnix(input.start ?? input.start_time ?? input.startTs ?? input.ts_start ?? input.time) ?? fallbackStart;
      const end = toUnix(input.end ?? input.end_time ?? input.endTs ?? input.ts_end) ?? fallbackEnd;
      const topRaw = num(input.top ?? input.high ?? input.price_high, NaN);
      const bottomRaw = num(input.bottom ?? input.low ?? input.price_low, NaN);
      if (!Number.isFinite(topRaw) || !Number.isFinite(bottomRaw)) return null;
      const top = Math.max(topRaw, bottomRaw);
      const bottom = Math.min(topRaw, bottomRaw);
      const zone: MarketZone = {
        kind,
        side,
        start,
        end: end < start ? start + 15 * 60 * 20 : end,
        top,
        bottom,
        label: String(input.label ?? `${side} ${kind}`),
      };
      const parsedScore = num(input.score, NaN);
      if (Number.isFinite(parsedScore)) {
        zone.score = parsedScore;
      }
      return zone;
    })
    .filter((zone): zone is MarketZone => zone !== null);

  return parsed.length ? parsed.slice(-24) : derived;
}

function buildReasoningEnvelope(payload: Record<string, unknown>, event: Record<string, unknown> = {}): ReasoningTree {
  const nested = payload.reasoning;
  return {
    event: {
      type: String(event.event_type || event.type || event.event || "signal"),
      trade_id: String(event.trade_id || payload.trade_id || ""),
      timestamp: String(event.timestamp || payload.timestamp || ""),
    },
    decision: {
      asset: String(payload.asset || event.asset || "XBTUSD"),
      side: String(payload.side || payload.direction || "long"),
      tier: String(payload.tier || "A"),
      confluence: num(payload.confluence || payload.conf || payload.conf_score, 0.7),
      evr: num(payload.evr, 1.8),
      risk_mode: String(payload.risk_mode || "normal"),
      hedge_ratio: num(payload.hedge_ratio, 0),
      regime: String(payload.regime || payload.regime_state || "unknown"),
      reason: String(payload.reason || payload.detail || event.event_type || "signal"),
    },
    reasoning:
      nested && typeof nested === "object"
        ? (nested as ReasoningTree)
        : {
            ml: {
              p_liq_flow: num(payload.p_liq_flow || payload.prob_liq_flow, 0.72),
              p_bos_cont: num(payload.p_bos_cont || payload.prob_bos_cont, 0.68),
              p_flow_1h: num(payload.p_flow_1h || payload.prob_flow_1h || payload.flow_1h, 0.64),
              hazard_score: num(payload.hazard || payload.hazard_score, 0.22),
            },
            context: {
              regime: String(payload.regime || payload.regime_state || "unknown"),
              session: String(payload.session || "unknown"),
            },
          },
  };
}

function makeDemoSnapshot(): TerminalSnapshot {
  const demoSignals: SignalCandidate[] = [
    {
      id: "SIG-1042",
      asset: "BTCUSD",
      side: "long",
      tier: "A+",
      confluence: 0.88,
      evr: 2.7,
      flow1h: 0.74,
      hazard: 0.18,
      regime: "trend",
      reason: "6h structure + 1h pulse + clean 15m retest",
      reasoning: {
        event: { type: "entry", trade_id: "SIG-1042", timestamp: "2026-03-03T10:15:00Z" },
        decision: { asset: "BTCUSD", side: "long", tier: "A+", confluence: 0.88, evr: 2.7, risk_mode: "normal", hedge_ratio: 0, regime: "trend", reason: "6h structure + 1h pulse + clean 15m retest" },
        reasoning: {
          ml: { p_liq_flow: 0.81, p_bos_cont: 0.78, p_flow_1h: 0.74, prob_confluence: 0.88 },
          smc: {
            bos: { bos_up: 1, choch: 0 },
            sweeps: { sweep_high: 1, sweep_low: 0 },
            zones: { zone_score_6h: 0.84, demand_zone: true, supply_zone: false },
          },
          flow: { flow_strength_1h: 0.79, displacement_body_pct_1h: 0.68, volume_z_1h: 1.42 },
          regime: { regime_state: "trend_expansion", p_regime_trend: 0.77, p_regime_expansion: 0.61 },
          hazard: { hazard_score: 0.18 },
          final_decision: { tier: "A+", confluence: 0.88, evr: 2.7, median_r: 4.6 },
        },
      },
    },
    {
      id: "SIG-1043",
      asset: "ETHUSD",
      side: "long",
      tier: "A",
      confluence: 0.82,
      evr: 2.2,
      flow1h: 0.69,
      hazard: 0.22,
      regime: "trend",
      reason: "VWAP deformation recovered with supportive liquidity distance",
      reasoning: buildReasoningEnvelope({ asset: "ETHUSD", tier: "A", confluence: 0.82, evr: 2.2, flow_1h: 0.69, hazard: 0.22, regime: "trend", reason: "VWAP deformation recovered with supportive liquidity distance" }),
    },
    {
      id: "SIG-1044",
      asset: "SOLUSD",
      side: "long",
      tier: "A",
      confluence: 0.79,
      evr: 2.0,
      flow1h: 0.66,
      hazard: 0.24,
      regime: "trend",
      reason: "Displacement confirmed, but session overlap not yet fully engaged",
      reasoning: buildReasoningEnvelope({ asset: "SOLUSD", tier: "A", confluence: 0.79, evr: 2.0, flow_1h: 0.66, hazard: 0.24, regime: "trend", reason: "Displacement confirmed, but session overlap not yet fully engaged" }),
    },
    {
      id: "SIG-1045",
      asset: "LINKUSD",
      side: "short",
      tier: "B",
      confluence: 0.71,
      evr: 1.8,
      flow1h: 0.58,
      hazard: 0.31,
      regime: "range",
      reason: "Counter-bias fade only, lower rank",
      reasoning: buildReasoningEnvelope({ asset: "LINKUSD", side: "short", tier: "B", confluence: 0.71, evr: 1.8, flow_1h: 0.58, hazard: 0.31, regime: "range", reason: "Counter-bias fade only, lower rank" }),
    },
    {
      id: "SIG-1046",
      asset: "XRPUSD",
      side: "long",
      tier: "B",
      confluence: 0.68,
      evr: 1.6,
      flow1h: 0.56,
      hazard: 0.29,
      regime: "compression",
      reason: "Structure valid but opportunity surface still compressed",
      reasoning: buildReasoningEnvelope({ asset: "XRPUSD", tier: "B", confluence: 0.68, evr: 1.6, flow_1h: 0.56, hazard: 0.29, regime: "compression", reason: "Structure valid but opportunity surface still compressed" }),
    },
  ];
  const demoTrades: AuditTrade[] = [
    {
      tradeId: "TR-3112",
      asset: "BTCUSD",
      side: "long",
      leg: "core",
      tier: "A+",
      pnl: 840,
      r: 3.0,
      entryPrice: 62110,
      exitPrice: 62890,
      qty: 0.32,
      notional: 20000,
      riskUsd: 280,
      fees: 14,
      slippageBps: 1.8,
      holdMinutes: 150,
      status: "closed",
      model: "confluence_model",
      reason: "core_tp_3.0R",
      entryTs: "2026-03-03T10:30:00Z",
      exitTs: "2026-03-03T13:00:00Z",
    },
    {
      tradeId: "TR-3113",
      asset: "BTCUSD",
      side: "long",
      leg: "runner",
      tier: "A+",
      pnl: 1560,
      r: 7.2,
      entryPrice: 62110,
      exitPrice: 63640,
      qty: 0.16,
      notional: 10000,
      riskUsd: 140,
      fees: 11,
      slippageBps: 2.2,
      holdMinutes: 315,
      status: "closed",
      model: "confluence_model",
      reason: "runner_tp_7.2R",
      entryTs: "2026-03-03T10:30:00Z",
      exitTs: "2026-03-03T15:45:00Z",
    },
    {
      tradeId: "TR-3118",
      asset: "ETHUSD",
      side: "long",
      leg: "core",
      tier: "A",
      pnl: 420,
      r: 2.0,
      entryPrice: 3412,
      exitPrice: 3471,
      qty: 2.9,
      notional: 9894,
      riskUsd: 210,
      fees: 9,
      slippageBps: 2.9,
      holdMinutes: 90,
      status: "closed",
      model: "meta_model",
      reason: "core_tp_2.0R",
      entryTs: "2026-03-03T12:00:00Z",
      exitTs: "2026-03-03T13:30:00Z",
    },
  ];

  return {
    meta: {
      source: "demo",
      lastUpdated: new Date().toISOString(),
      repoRoot: REPO_ROOT,
      modelVersion: "demo-v1",
      transport: "fastapi + websocket preferred, artifact fallback available",
      viewModeRequested: "auto",
      viewModeEffective: "auto",
    },
    mission: {
      headline: "Terminal primed for deterministic execution parity",
      status: "Monitoring",
      substatus: "No live artifacts found yet, serving research-grade demo state.",
      metrics: [
        { label: "Cycle Capital", value: "$20,000", tone: "amber", delta: "base ticket" },
        { label: "Deployable", value: "$21,240", tone: "teal", delta: "+6.2%" },
        { label: "Locked Profit", value: "$1,240", tone: "cyan", delta: "vaulted" },
        { label: "Open Positions", value: "2", tone: "slate", delta: "core + runner" },
        { label: "Cooling", value: "Inactive", tone: "teal", delta: "eligible" },
      ],
    },
    insights: {
      summary: "The system is reading the market as trend-persistent with constructive 1h flow and non-fragile liquidity posture.",
      trace: [
        { label: "Structural Bias", value: "Bullish 6h", detail: "Zone score and premium/discount remain aligned with upward continuation.", tone: "cyan" },
        { label: "Liquidity Geometry", value: "Sweep repaired", detail: "Recent equal-high sweep has been structurally repaired without fresh CHOCH failure.", tone: "teal" },
        { label: "Flow Pulse", value: "0.74", detail: "1h flow model is confirming displacement freshness above the continuation threshold.", tone: "amber" },
        { label: "Execution Eligibility", value: "Pass", detail: "Volatility, session weight, and hazard posture remain inside modeled tolerances.", tone: "teal" },
      ],
      latestReasoning: demoSignals[0].reasoning,
    },
    regime: {
      current: "Trend Expansion",
      persistence: 81,
      transitionRisk: 19,
      states: [
        { name: "Trend Expansion", probability: 0.62, description: "Directional persistence with healthy liquidity participation." },
        { name: "Trend Compression", probability: 0.21, description: "Trend still intact, but range contraction is building." },
        { name: "Range Mean-Revert", probability: 0.11, description: "Lower expectancy for continuation setups." },
        { name: "Stress Breakdown", probability: 0.06, description: "Macro / liquidity instability would suspend risk." },
      ],
    },
    signals: {
      summary: "Five ranked candidates are kept visible so operators can compare coherence, not just raw signal strength.",
      candidates: demoSignals,
    },
    risk: {
      summary: "Risk radar is green overall, with slight caution on slippage concentration during overlap transitions.",
      stress: 24,
      slippage: 33,
      exposure: 41,
      guardrails: [
        { label: "Macro Constraint Gate", status: "pass", detail: "No macro dislocation or volatility shock detected." },
        { label: "Liquidity Degradation", status: "warn", detail: "Book depth is thinner around overlap rotation." },
        { label: "Execution Feasibility", status: "pass", detail: "Expected impact remains inside the modeled error surface." },
        { label: "Cooling Logic", status: "pass", detail: "Compounding cycle remains active; no vault reset required." },
      ],
    },
    performance: buildPerformance(demoTrades),
    market: buildMarket(demoTrades, demoSignals),
    audit: {
      summary: "Every state shown here is meant to map cleanly back to the same deterministic feature graph used in research and execution.",
      trades: demoTrades,
      events: [
        { timestamp: "2026-03-03T10:15:00Z", type: "scanner", detail: "BTCUSD ranked A+ with confluence 0.88 and EVR 2.7." },
        { timestamp: "2026-03-03T10:30:00Z", type: "entry", detail: "Core and runner legs opened with base cycle capital sizing." },
        { timestamp: "2026-03-03T11:45:00Z", type: "risk", detail: "Stop moved to breakeven after 2R ladder threshold." },
        { timestamp: "2026-03-03T15:45:00Z", type: "exit", detail: "Runner closed after extended continuation target." },
      ],
    },
  };
}

function deriveGuardrails(state: Record<string, unknown>): Guardrail[] {
  const cooling = Boolean(state.cooling_to);
  const drawdown = num(state.max_drawdown);
  return [
    {
      label: "Cooling Logic",
      status: cooling ? "warn" : "pass",
      detail: cooling ? `Cooling active until ${String(state.cooling_to)}` : "Compounding cycle is active.",
    },
    {
      label: "Drawdown Surface",
      status: drawdown > 10 ? "block" : drawdown > 4 ? "warn" : "pass",
      detail: `Current drawdown ${fmtPct(drawdown)}.`,
    },
    {
      label: "Execution Readiness",
      status: num(state.open_positions) > 4 ? "warn" : "pass",
      detail: `${num(state.open_positions)} positions currently open.`,
    },
  ];
}

function buildSignalsFromEvents(events: Record<string, string>[]): SignalCandidate[] {
  const entries = events.filter((row) => ["entry", "signal", "scanner"].includes(String(row.type || row.event || "").toLowerCase()));
  if (!entries.length) {
    return makeDemoSnapshot().signals.candidates;
  }
  return entries.slice(-5).reverse().map((row, idx) => ({
    id: String(row.trade_id || row.id || `SIG-${idx + 1}`),
    asset: String(row.asset || "XBTUSD"),
    side: String(row.side || "long").toLowerCase() === "short" ? "short" : "long",
    tier: String(row.tier || "A"),
    confluence: num(row.confluence || row.conf || row.score, 0.7),
    evr: num(row.evr, 1.8),
    flow1h: num(row.flow_1h || row.prob_flow_1h || row.p_flow_1h, 0.62),
    hazard: num(row.hazard || row.hazard_score, 0.22),
    regime: String(row.regime || row.regime_state || "unknown"),
    reason: String(row.reason || row.detail || row.event || "Derived from live event stream."),
    reasoning: buildReasoningEnvelope(row, row),
  }));
}

function buildTrades(rows: Record<string, string>[]): AuditTrade[] {
  return rows.slice(-120).reverse().map((row, idx) => {
    const entryTs = String(row.entry_ts || row.entry_time || row.timestamp || "");
    const exitTs = String(row.exit_ts || row.exit_time || "");
    const entryMs = entryTs ? Date.parse(entryTs) : NaN;
    const exitMs = exitTs ? Date.parse(exitTs) : NaN;
    const holdMinutes = Number.isFinite(entryMs) && Number.isFinite(exitMs)
      ? Math.max(0, Math.round((exitMs - entryMs) / 60000))
      : num(row.hold_minutes, 0);
    const side = String(row.side || row.direction || "long").toLowerCase() === "short" ? "short" : "long";
    const status = exitTs ? "closed" : "open";
    return {
      tradeId: String(row.trade_id || `TR-${idx + 1}`),
      asset: String(row.asset || "XBTUSD"),
      side,
      leg: String(row.leg || "core"),
      tier: String(row.tier || "unranked"),
      pnl: num(row.pnl),
      r: num(row.r),
      session: String(row.session || row.session_name || "unknown"),
      regime: String(row.regime || row.regime_state || "unknown"),
      entryPrice: num(row.entry_price),
      exitPrice: num(row.exit_price),
      qty: num(row.qty || row.quantity),
      notional: num(row.notional || row.notional_usd || row.position_notional),
      riskUsd: num(row.risk_usd || row.risk),
      fees: num(row.fees || row.fee_usd || row.total_fees),
      slippageBps: num(row.slippage_bps || row.slip_bps),
      mae: num(row.mae || row.mae_r || row.max_adverse_excursion),
      mfe: num(row.mfe || row.mfe_r || row.max_favorable_excursion),
      holdMinutes,
      status,
      model: String(row.model || row.model_name || row.tier || "multi"),
      reason: String(row.reason || row.result || "closed"),
      entryTs,
      exitTs,
    };
  });
}

function buildEvents(rows: Record<string, string>[]): AuditEvent[] {
  return rows.slice(-8).reverse().map((row, idx) => ({
    timestamp: String(row.timestamp || row.entry_ts || row.exit_ts || new Date(Date.now() - idx * 60000).toISOString()),
    type: String(row.type || row.event || row.reason || "event"),
    detail: String(row.detail || row.reason || row.asset || row.trade_id || "Replayable event"),
  }));
}

function buildPerformance(trades: AuditTrade[]): TerminalSnapshot["performance"] {
  const closedTrades = trades.filter((trade) => trade.status !== "open");
  const scope = closedTrades.length ? closedTrades : trades;
  const ordered = [...scope].sort((a, b) => {
    const aTs = Date.parse(a.exitTs || a.entryTs || "");
    const bTs = Date.parse(b.exitTs || b.entryTs || "");
    if (!Number.isFinite(aTs) && !Number.isFinite(bTs)) return 0;
    if (!Number.isFinite(aTs)) return 1;
    if (!Number.isFinite(bTs)) return -1;
    return aTs - bTs;
  });
  const netPnl = scope.reduce((acc, trade) => acc + num(trade.pnl), 0);
  const grossProfit = scope.reduce((acc, trade) => acc + Math.max(0, num(trade.pnl)), 0);
  const grossLoss = scope.reduce((acc, trade) => acc + Math.min(0, num(trade.pnl)), 0);
  const wins = scope.filter((trade) => num(trade.pnl) > 0).length;
  const winRate = scope.length ? (wins / scope.length) * 100 : 0;
  const avgR = scope.length ? scope.reduce((acc, trade) => acc + num(trade.r), 0) / scope.length : 0;
  const avgHold = scope.length
    ? scope.reduce((acc, trade) => acc + num(trade.holdMinutes), 0) / scope.length
    : 0;
  const feesTotal = scope.reduce((acc, trade) => acc + Math.max(0, num(trade.fees)), 0);
  const avgSlippage = scope.length
    ? scope.reduce((acc, trade) => acc + Math.max(0, num(trade.slippageBps)), 0) / scope.length
    : 0;
  const profitFactor = grossLoss < 0 ? grossProfit / Math.abs(grossLoss) : grossProfit > 0 ? 99 : 0;
  const maxLoss = scope.reduce((acc, trade) => Math.min(acc, num(trade.pnl)), 0);
  const pnlValues = scope.map((trade) => num(trade.pnl));
  const rValues = scope.map((trade) => num(trade.r));
  const sortedPnl = [...pnlValues].sort((a, b) => a - b);
  const sortedR = [...rValues].sort((a, b) => a - b);
  const winsOnly = pnlValues.filter((value) => value > 0);
  const lossesOnly = pnlValues.filter((value) => value < 0);
  const avgWin = winsOnly.length ? winsOnly.reduce((a, b) => a + b, 0) / winsOnly.length : 0;
  const avgLoss = lossesOnly.length ? lossesOnly.reduce((a, b) => a + b, 0) / lossesOnly.length : 0;
  const payoffRatio = avgLoss < 0 ? avgWin / Math.abs(avgLoss) : 0;
  const median = (arr: number[]) => {
    if (!arr.length) return 0;
    const mid = Math.floor(arr.length / 2);
    return arr.length % 2 === 0 ? (arr[mid - 1] + arr[mid]) / 2 : arr[mid];
  };

  let maxConsecutiveWins = 0;
  let maxConsecutiveLosses = 0;
  let winRun = 0;
  let lossRun = 0;
  for (const trade of ordered) {
    const pnl = num(trade.pnl);
    if (pnl > 0) {
      winRun += 1;
      lossRun = 0;
    } else if (pnl < 0) {
      lossRun += 1;
      winRun = 0;
    } else {
      winRun = 0;
      lossRun = 0;
    }
    maxConsecutiveWins = Math.max(maxConsecutiveWins, winRun);
    maxConsecutiveLosses = Math.max(maxConsecutiveLosses, lossRun);
  }

  let runningEquity = 20_000;
  let peakEquity = runningEquity;
  const equityTimeline = ordered.map((trade, idx) => {
    runningEquity += num(trade.pnl);
    peakEquity = Math.max(peakEquity, runningEquity);
    const drawdown = runningEquity - peakEquity;
    const ts = trade.exitTs || trade.entryTs || "";
    const parsed = ts ? Date.parse(ts) : NaN;
    const label = Number.isFinite(parsed)
      ? new Date(parsed).toISOString().slice(0, 16).replace("T", " ")
      : `trade-${idx + 1}`;
    return {
      label,
      ts,
      pnl: num(trade.pnl),
      trades: idx + 1,
      equity: runningEquity,
      drawdown,
    };
  });

  const timelineBucket = (period: "daily" | "monthly") => {
    const map = new Map<string, { pnl: number; trades: number; wins: number; rSum: number; ts: string }>();
    for (const trade of ordered) {
      const tsRaw = trade.exitTs || trade.entryTs || "";
      const parsed = tsRaw ? Date.parse(tsRaw) : NaN;
      if (!Number.isFinite(parsed)) continue;
      const dt = new Date(parsed);
      const year = dt.getUTCFullYear();
      const month = `${dt.getUTCMonth() + 1}`.padStart(2, "0");
      const day = `${dt.getUTCDate()}`.padStart(2, "0");
      const key = period === "daily" ? `${year}-${month}-${day}` : `${year}-${month}`;
      const ts = period === "daily" ? key : `${key}-01`;
      const prev = map.get(key) ?? { pnl: 0, trades: 0, wins: 0, rSum: 0, ts };
      prev.pnl += num(trade.pnl);
      prev.trades += 1;
      if (num(trade.pnl) > 0) prev.wins += 1;
      prev.rSum += num(trade.r);
      map.set(key, prev);
    }
    return [...map.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([label, stat]) => ({
        label,
        ts: stat.ts,
        pnl: stat.pnl,
        trades: stat.trades,
        winRate: stat.trades ? (stat.wins / stat.trades) * 100 : 0,
        avgR: stat.trades ? stat.rSum / stat.trades : 0,
      }));
  };

  const periodDefs = [
    { label: "Daily", ms: 24 * 60 * 60 * 1000 },
    { label: "Weekly", ms: 7 * 24 * 60 * 60 * 1000 },
    { label: "Monthly", ms: 30 * 24 * 60 * 60 * 1000 },
  ] as const;
  const nowMs = Date.now();
  const periods = periodDefs.map((period) => {
    const filtered = scope.filter((trade) => {
      const ts = trade.exitTs || trade.entryTs;
      const parsed = ts ? Date.parse(ts) : NaN;
      return Number.isFinite(parsed) && parsed >= nowMs - period.ms;
    });
    const pnl = filtered.reduce((acc, trade) => acc + num(trade.pnl), 0);
    const wr = filtered.length ? (filtered.filter((trade) => num(trade.pnl) > 0).length / filtered.length) * 100 : 0;
    const periodAvgR = filtered.length ? filtered.reduce((acc, trade) => acc + num(trade.r), 0) / filtered.length : 0;
    return {
      label: period.label,
      pnl,
      trades: filtered.length,
      winRate: wr,
      avgR: periodAvgR,
    };
  });

  const bucket = (selector: (trade: AuditTrade) => string): TerminalSnapshot["performance"]["byAsset"] => {
    const map = new Map<string, { pnl: number; trades: number; wins: number }>();
    for (const trade of scope) {
      const key = selector(trade) || "unknown";
      const prev = map.get(key) ?? { pnl: 0, trades: 0, wins: 0 };
      prev.pnl += num(trade.pnl);
      prev.trades += 1;
      if (num(trade.pnl) > 0) prev.wins += 1;
      map.set(key, prev);
    }
    return [...map.entries()]
      .map(([label, stat]) => ({
        label,
        pnl: stat.pnl,
        trades: stat.trades,
        winRate: stat.trades ? (stat.wins / stat.trades) * 100 : 0,
      }))
      .sort((a, b) => b.pnl - a.pnl)
      .slice(0, 8);
  };

  const weekdayOrder = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const byWeekday = (() => {
    const map = new Map<string, { pnl: number; trades: number; wins: number }>();
    for (const label of weekdayOrder) map.set(label, { pnl: 0, trades: 0, wins: 0 });
    for (const trade of ordered) {
      const parsed = Date.parse(trade.exitTs || trade.entryTs || "");
      if (!Number.isFinite(parsed)) continue;
      const idx = (new Date(parsed).getUTCDay() + 6) % 7;
      const key = weekdayOrder[idx];
      const slot = map.get(key)!;
      slot.pnl += num(trade.pnl);
      slot.trades += 1;
      if (num(trade.pnl) > 0) slot.wins += 1;
    }
    return weekdayOrder.map((label) => {
      const slot = map.get(label)!;
      return {
        label,
        pnl: slot.pnl,
        trades: slot.trades,
        winRate: slot.trades ? (slot.wins / slot.trades) * 100 : 0,
      };
    });
  })();

  const byHour = (() => {
    const map = new Map<string, { pnl: number; trades: number; wins: number }>();
    for (let hour = 0; hour < 24; hour += 1) {
      map.set(`${String(hour).padStart(2, "0")}:00`, { pnl: 0, trades: 0, wins: 0 });
    }
    for (const trade of ordered) {
      const parsed = Date.parse(trade.exitTs || trade.entryTs || "");
      if (!Number.isFinite(parsed)) continue;
      const hour = new Date(parsed).getUTCHours();
      const key = `${String(hour).padStart(2, "0")}:00`;
      const slot = map.get(key)!;
      slot.pnl += num(trade.pnl);
      slot.trades += 1;
      if (num(trade.pnl) > 0) slot.wins += 1;
    }
    return [...map.entries()].map(([label, stat]) => ({
      label,
      pnl: stat.pnl,
      trades: stat.trades,
      winRate: stat.trades ? (stat.wins / stat.trades) * 100 : 0,
    }));
  })();

  const byHold = (() => {
    const defs = [
      { label: "<15m", min: 0, max: 15 },
      { label: "15m-1h", min: 15, max: 60 },
      { label: "1h-4h", min: 60, max: 240 },
      { label: "4h-12h", min: 240, max: 720 },
      { label: ">12h", min: 720, max: Number.POSITIVE_INFINITY },
    ];
    const map = new Map<string, { pnl: number; trades: number; wins: number }>();
    for (const def of defs) map.set(def.label, { pnl: 0, trades: 0, wins: 0 });
    for (const trade of ordered) {
      const hold = Math.max(0, num(trade.holdMinutes));
      const bucketDef = defs.find((def) => hold >= def.min && hold < def.max);
      if (!bucketDef) continue;
      const slot = map.get(bucketDef.label)!;
      slot.pnl += num(trade.pnl);
      slot.trades += 1;
      if (num(trade.pnl) > 0) slot.wins += 1;
    }
    return defs.map((def) => {
      const stat = map.get(def.label)!;
      return {
        label: def.label,
        pnl: stat.pnl,
        trades: stat.trades,
        winRate: stat.trades ? (stat.wins / stat.trades) * 100 : 0,
      };
    });
  })();

  const ranked = [...scope].sort((a, b) => num(b.pnl) - num(a.pnl));
  const topWinners = ranked.slice(0, 10);
  const topLosers = [...ranked].reverse().slice(0, 10);
  const maxDrawdown = Math.abs(
    equityTimeline.reduce((acc, point) => Math.min(acc, num(point.drawdown)), 0)
  );

  return {
    summary: "Performance intelligence is synchronized with the same trade ledger used by runtime and audit views.",
    kpis: [
      { label: "Net PnL", value: fmtMoney(netPnl), tone: netPnl >= 0 ? "teal" : "rose", delta: `${scope.length} closed trades` },
      { label: "Win Rate", value: fmtPct(winRate), tone: winRate >= 55 ? "teal" : winRate >= 45 ? "amber" : "rose", delta: `${wins}/${scope.length || 0}` },
      { label: "Avg R", value: fmtR(avgR), tone: avgR >= 0 ? "cyan" : "rose", delta: "per trade" },
      { label: "Profit Factor", value: profitFactor >= 99 ? "N/A" : profitFactor.toFixed(2), tone: profitFactor >= 1.3 ? "teal" : profitFactor >= 1.0 ? "amber" : "rose", delta: "gross win / gross loss" },
      { label: "Max Loss", value: fmtMoney(maxLoss), tone: "rose", delta: "single trade" },
      { label: "Avg Hold", value: `${Math.round(avgHold)}m`, tone: "slate", delta: "duration" },
      { label: "Fees", value: fmtMoney(feesTotal), tone: "amber", delta: `avg slip ${avgSlippage.toFixed(2)} bps` },
    ],
    periods,
    byAsset: bucket((trade) => trade.asset),
    byTier: bucket((trade) => trade.tier),
    byModel: bucket((trade) => trade.model || "unknown"),
    bySession: bucket((trade) => trade.session || "unknown"),
    byRegime: bucket((trade) => trade.regime || "unknown"),
    byWeekday,
    byHour,
    byHold,
    topWinners,
    topLosers,
    expectancy: {
      expectancyR: avgR,
      avgWin,
      avgLoss,
      payoffRatio,
      medianPnl: median(sortedPnl),
      medianR: median(sortedR),
      maxConsecutiveWins,
      maxConsecutiveLosses,
      maxDrawdown,
    },
    timeline: {
      equity: equityTimeline.slice(-500),
      daily: timelineBucket("daily").slice(-180),
      monthly: timelineBucket("monthly").slice(-48),
    },
    tradeTable: trades.slice(0, 260),
  };
}

function buildMarket(
  trades: AuditTrade[],
  signals: SignalCandidate[],
  candlePayload?: unknown,
): TerminalSnapshot["market"] {
  const payloadRecord = candlePayload && typeof candlePayload === "object"
    ? (candlePayload as Record<string, unknown>)
    : null;
  const candles = (() => {
    const parsed = parseCandlePayload(payloadRecord?.candles ?? payloadRecord?.m15 ?? payloadRecord?.["15m"] ?? candlePayload);
    if (parsed.length) return parsed;
    const anchor = trades[0]?.exitPrice || trades[0]?.entryPrice || 100_000;
    return demoCandles(anchor);
  })();
  const timeframes = parseTimeframesPayload(payloadRecord?.timeframes ?? payloadRecord, candles) ?? buildTimeframes(candles);
  const primary = timeframes.m15.length ? timeframes.m15 : candles;

  const markers: TerminalSnapshot["market"]["markers"] = [];
  for (const trade of trades.slice(0, 80)) {
    const side = trade.side === "short" ? "short" : "long";
    const entry = toUnix(trade.entryTs);
    if (entry) {
      markers.push({
        time: entry,
        position: side === "long" ? "belowBar" : "aboveBar",
        color: side === "long" ? "#2ae6b8" : "#ff6b88",
        shape: side === "long" ? "arrowUp" : "arrowDown",
        text: `entry ${trade.tier}`,
      });
    }
    const exit = toUnix(trade.exitTs);
    if (exit) {
      markers.push({
        time: exit,
        position: side === "long" ? "aboveBar" : "belowBar",
        color: trade.pnl >= 0 ? "#2ae6b8" : "#ff6b88",
        shape: "circle",
        text: `exit ${trade.pnl >= 0 ? "+" : ""}${trade.pnl.toFixed(0)}`,
      });
    }
  }
  for (const signal of signals.slice(0, 6)) {
    const ts = toUnix(
      (signal.reasoning as Record<string, unknown> | undefined)?.event
        ? String(((signal.reasoning as Record<string, unknown>).event as Record<string, unknown>).timestamp || "")
        : "",
    );
    if (!ts) continue;
    markers.push({
      time: ts,
      position: "inBar",
      color: "#f6b63c",
      shape: "square",
      text: `signal ${signal.tier}`,
    });
  }
  const zones = parseZonesPayload(payloadRecord?.zones, primary, signals);

  const last = primary[primary.length - 1];
  const lookback = primary.slice(-96);
  const first = lookback[0] ?? last;
  const changePct = first?.close ? ((last.close / first.close - 1) * 100) : 0;
  const high = lookback.reduce((acc, row) => Math.max(acc, row.high), Number.NEGATIVE_INFINITY);
  const low = lookback.reduce((acc, row) => Math.min(acc, row.low), Number.POSITIVE_INFINITY);
  const rangePct = Number.isFinite(high) && Number.isFinite(low) && last?.close
    ? ((high - low) / last.close) * 100
    : 0;
  const returns = primary.slice(1).map((row, idx) => row.close / primary[idx].close - 1);
  const mean = returns.length ? returns.reduce((acc, row) => acc + row, 0) / returns.length : 0;
  const variance = returns.length
    ? returns.reduce((acc, row) => acc + (row - mean) ** 2, 0) / returns.length
    : 0;
  const vol = Math.sqrt(Math.max(0, variance)) * 100;
  const vol24 = lookback.reduce((acc, row) => acc + row.volume, 0);

  const activeTrades = trades.filter((trade) => trade.status === "open");

  return {
    symbol: signals[0]?.asset || trades[0]?.asset || "BTCUSD",
    timeframe: "15m",
    summary: "Unified TradingView-style canvas with synchronized 12h/6h/1h/15m context, SMC overlays, and replay-ready trade lifecycle markers.",
    candles: primary,
    markers: markers.slice(-220),
    zones,
    timeframes,
    stats: [
      { label: "Last Price", value: fmtMoney(last?.close ?? 0), tone: "cyan", detail: "latest 15m close" },
      { label: "24h Change", value: `${changePct >= 0 ? "+" : ""}${changePct.toFixed(2)}%`, tone: changePct >= 0 ? "teal" : "rose", detail: "96 bars lookback" },
      { label: "24h Range", value: `${rangePct.toFixed(2)}%`, tone: "amber", detail: "high-low span" },
      { label: "Realized Vol", value: vol.toFixed(3), tone: "amber", detail: "std(ret) %" },
      { label: "24h Volume", value: vol24.toFixed(0), tone: "teal", detail: "sum(volume)" },
      { label: "Markers", value: String(markers.slice(-220).length), tone: "slate", detail: "entry/exit/signal" },
      { label: "Zones", value: String(zones.length), tone: "cyan", detail: "OB/FVG/liquidity overlays" },
    ],
    activeTrades: (activeTrades.length ? activeTrades : trades).slice(0, 12),
  };
}

function withPerformance(snapshot: TerminalSnapshot): TerminalSnapshot {
  if (snapshot.performance) {
    return snapshot;
  }
  const trades = snapshot.audit?.trades ?? [];
  return {
    ...snapshot,
    performance: buildPerformance(trades),
  };
}

function withMarket(snapshot: TerminalSnapshot): TerminalSnapshot {
  const trades = snapshot.audit?.trades ?? [];
  const signals = snapshot.signals?.candidates ?? [];
  if (snapshot.market && snapshot.market.candles.length) {
    const refreshed = buildMarket(trades, signals, snapshot.market);
    return {
      ...snapshot,
      market: {
        ...refreshed,
        ...snapshot.market,
        candles: snapshot.market.candles?.length ? snapshot.market.candles : refreshed.candles,
        markers: snapshot.market.markers?.length ? snapshot.market.markers : refreshed.markers,
        zones: snapshot.market.zones?.length ? snapshot.market.zones : refreshed.zones,
        timeframes: snapshot.market.timeframes?.m15?.length
          ? snapshot.market.timeframes
          : refreshed.timeframes,
        stats: snapshot.market.stats?.length ? snapshot.market.stats : refreshed.stats,
        activeTrades: snapshot.market.activeTrades?.length ? snapshot.market.activeTrades : refreshed.activeTrades,
      },
    };
  }
  return {
    ...snapshot,
    market: buildMarket(trades, signals),
  };
}

export async function loadTerminalSnapshot(mode: TerminalMode = "auto"): Promise<TerminalSnapshot> {
  const requestedMode = normalizeMode(mode, "auto");

  if (BACKEND_API_URL && (requestedMode === "auto" || requestedMode === "live")) {
    try {
      const url = new URL(BACKEND_API_URL);
      url.searchParams.set("mode", requestedMode);
      const response = await fetch(url, { cache: "no-store" });
      if (response.ok) {
        const snapshot = withMarket(withPerformance((await response.json()) as TerminalSnapshot));
        return {
          ...snapshot,
          meta: {
            ...snapshot.meta,
            source: "telemetry",
            viewModeRequested: requestedMode,
            viewModeEffective: requestedMode === "auto" ? "live" : requestedMode,
          },
        };
      }
    } catch {
      // fall back to local artifact loader
    }
  }

  const backtestRoot = await chooseBundleRoot(
    uniqueRoots([BACKTEST_DIR_OVERRIDE, ...BACKTEST_DIR_CANDIDATES]),
    BACKTEST_FILE_MARKERS,
  );
  const forwardRoot = await chooseBundleRoot(
    uniqueRoots([FORWARD_DIR_OVERRIDE, ...FORWARD_DIR_CANDIDATES]),
    RUNTIME_FILE_MARKERS,
  );
  const liveRoot = await chooseBundleRoot(
    uniqueRoots([LIVE_DIR_OVERRIDE, ...LIVE_DIR_CANDIDATES]),
    RUNTIME_FILE_MARKERS,
  );
  const modelRoot = path.join(REPO_ROOT, "models");

  const backtestSummary = await readFirstJson<Record<string, unknown>>(backtestRoot, ["summary.json"]);
  const backtestLedger = await readFirstCsv(backtestRoot, ["ledger.csv", "trades.csv"]);
  const backtestCandles = await readFirstCsv(backtestRoot, ["candles_15m.csv", "candles.csv"]);
  const backtestReasoning = await readFirstJson<Record<string, unknown>>(backtestRoot, ["reasoning.json"]);
  const backtestBundle: BacktestBundle = {
    root: backtestRoot,
    summary: backtestSummary,
    ledger: backtestLedger,
    candles: backtestCandles,
    reasoning: backtestReasoning,
    mtimeMs: Math.max(
      await statMtimeMs(backtestRoot ? path.join(backtestRoot, "summary.json") : ""),
      await statMtimeMs(backtestRoot ? path.join(backtestRoot, "ledger.csv") : ""),
      await statMtimeMs(backtestRoot ? path.join(backtestRoot, "trades.csv") : ""),
    ),
  };

  const loadRuntimeBundle = async (root: string | null): Promise<RuntimeBundle> => {
    const snapshot = await readFirstJson<Record<string, unknown>>(root, ["snapshot.json"]);
    const fallbackState = await readFirstJson<Record<string, unknown>>(root, ["state.json"]);
    const fallbackEventsJson = await readFirstJson<unknown[]>(root, ["events.json"]);
    const fallbackEventsCsv = await readFirstCsv(root, ["events.csv"]);
    const fallbackLedger = await readFirstCsv(root, ["closed_trades.csv"]);
    const fallbackCandles = await readFirstCsv(root, ["candles.csv", "bars.csv"]);
    const snapshotState = snapshot?.state && typeof snapshot.state === "object"
      ? (snapshot.state as Record<string, unknown>)
      : null;
    const snapshotEvents = Array.isArray(snapshot?.events)
      ? (snapshot.events as unknown[]).map((row) => flattenPayloadRow(row))
      : [];
    const jsonEvents = Array.isArray(fallbackEventsJson)
      ? fallbackEventsJson.map((row) => flattenPayloadRow(row))
      : [];
    const state = snapshotState ?? fallbackState ?? {};
    const ledger = fallbackLedger.length ? fallbackLedger : stateClosedTrades(state);
    const candles = fallbackCandles.length
      ? fallbackCandles
      : Array.isArray(snapshot?.candles)
        ? (snapshot.candles as unknown[]).map((row) => flattenPayloadRow(row))
        : [];
    const mtimeMs = Math.max(
      await statMtimeMs(root ? path.join(root, "snapshot.json") : ""),
      await statMtimeMs(root ? path.join(root, "state.json") : ""),
      await statMtimeMs(root ? path.join(root, "events.json") : ""),
      await statMtimeMs(root ? path.join(root, "events.csv") : ""),
      await statMtimeMs(root ? path.join(root, "closed_trades.csv") : ""),
      await statMtimeMs(root ? path.join(root, "candles.csv") : ""),
      await statMtimeMs(root ? path.join(root, "bars.csv") : ""),
    );
    return {
      root,
      snapshot,
      state,
      events: snapshotEvents.length ? snapshotEvents : jsonEvents.length ? jsonEvents : fallbackEventsCsv,
      ledger,
      candles,
      mtimeMs,
    };
  };

  const forwardBundle = await loadRuntimeBundle(forwardRoot);
  const liveBundle = await loadRuntimeBundle(liveRoot);
  const effectiveMode = resolveEffectiveMode(requestedMode, forwardBundle, liveBundle);

  const runtimeBundle = effectiveMode === "forward"
    ? forwardBundle
    : effectiveMode === "live"
      ? liveBundle
      : null;
  const snapshot = runtimeBundle?.snapshot ?? null;
  const state = runtimeBundle?.state ?? {};
  const summary = effectiveMode === "backtest" ? backtestBundle.summary : null;
  const events = effectiveMode === "backtest" ? [] : runtimeBundle?.events ?? [];
  const ledger = effectiveMode === "backtest" ? backtestBundle.ledger : runtimeBundle?.ledger ?? [];
  const marketPayload = effectiveMode === "backtest"
    ? backtestBundle.candles
    : snapshot?.market ?? snapshot?.candles ?? runtimeBundle?.candles ?? null;

  const version = await listLatestModelVersion(modelRoot);
  if (!summary && !snapshot && !ledger.length && !events.length) {
    const demo = makeDemoSnapshot();
    return {
      ...demo,
      meta: {
        ...demo.meta,
        viewModeRequested: requestedMode,
        viewModeEffective: requestedMode === "auto" ? "backtest" : requestedMode,
      },
    };
  }

  const guardrails = deriveGuardrails(state);
  const equity = num(state.equity, num(summary?.ending_equity, 20_000));
  const freeCapital = num(state.free_capital, equity);
  const lockedProfit = num(state.locked_profit, 0);
  const openPositions = num(state.open_positions, 0);
  const trades = buildTrades(ledger);
  const closedTrades = trades.filter((trade) => trade.status !== "open");
  const signalSourceTrades = closedTrades.length ? closedTrades : trades;
  const wins = signalSourceTrades.filter((trade) => num(trade.pnl) > 0).length;
  const fallbackWinRate = signalSourceTrades.length ? (wins / signalSourceTrades.length) * 100 : 0;
  const summaryWinRate = num(summary?.win_rate, NaN);
  const winRate = Number.isFinite(summaryWinRate)
    ? summaryWinRate * (summaryWinRate <= 1 ? 100 : 1)
    : fallbackWinRate;
  const maxDrawdown = Math.abs(num(summary?.max_drawdown, num(state.max_drawdown, 0)));
  const signals = buildSignalsFromEvents(events);
  const performance = buildPerformance(trades);
  const market = buildMarket(
    trades,
    signals,
    marketPayload,
  );
  const latestReasoning = effectiveMode === "backtest"
    ? asReasoningTree(backtestBundle.reasoning)
    : signals[0]?.reasoning;

  return withMarket(withPerformance({
    meta: {
      source: "artifacts",
      lastUpdated: new Date().toISOString(),
      repoRoot: REPO_ROOT,
      modelVersion: version,
      transport: "fastapi + websocket preferred, artifact fallback available",
      viewModeRequested: requestedMode,
      viewModeEffective: effectiveMode,
    },
    mission: {
      headline: "Artifact-backed terminal state loaded from repaired repo outputs",
      status: state.cooling_to ? "Cooling" : openPositions ? "Active" : "Monitoring",
      substatus: state.cooling_to
        ? `Cooling active until ${String(state.cooling_to)}`
        : `${openPositions} positions visible through ${effectiveMode} artifacts.`,
      metrics: [
        { label: "Equity", value: fmtMoney(equity), tone: "cyan", delta: version },
        { label: "Free Capital", value: fmtMoney(freeCapital), tone: "teal", delta: "deployable" },
        { label: "Locked Profit", value: fmtMoney(lockedProfit), tone: "amber", delta: "vaulted" },
        { label: "Open Positions", value: String(openPositions), tone: openPositions ? "amber" : "slate", delta: `${effectiveMode} state` },
        { label: "Win Rate", value: fmtPct(winRate), tone: winRate >= 55 ? "teal" : "rose", delta: effectiveMode === "backtest" ? "backtest summary" : `${effectiveMode} ledger` },
      ],
    },
    insights: {
      summary: "Insights are currently derived from live state, recent events, and repaired model artifact metadata.",
      trace: [
        { label: "Capital Cycle", value: lockedProfit > 0 ? "Compounding" : "Base ticket", detail: lockedProfit > 0 ? "Profit has been vaulted while cycle capital remains active." : "System is trading from the base allocation.", tone: "amber" },
        { label: "Execution Posture", value: state.cooling_to ? "Guarded" : "Eligible", detail: state.cooling_to ? "Cooling timer is active under current state." : "No cooling blocker is present in state artifacts.", tone: state.cooling_to ? "rose" : "teal" },
        { label: "Model Surface", value: version, detail: "Latest discovered model registry version across repaired artifact directories.", tone: "cyan" },
        { label: "Decision Trace", value: `${events.length} events`, detail: "The frontend is reading the same persisted event stream the repaired runtime emits today.", tone: "teal" },
        {
          label: "Artifact Route",
          value: effectiveMode,
          detail:
            effectiveMode === "backtest"
              ? (backtestBundle.root ?? "no backtest artifacts discovered")
              : (runtimeBundle?.root ?? `no ${effectiveMode} artifacts discovered`),
          tone: effectiveMode === "live" ? "teal" : effectiveMode === "forward" ? "cyan" : "amber",
        },
      ],
      latestReasoning,
    },
    regime: {
      current: String((ledger[0]?.regime || events[0]?.regime || "unknown")).replaceAll("_", " "),
      persistence: Math.max(35, 100 - Math.round(maxDrawdown)),
      transitionRisk: Math.min(65, Math.round(maxDrawdown + num(state.open_positions) * 4)),
      states: [
        { name: "Current", probability: 0.58, description: "Dominant regime inferred from current artifacts." },
        { name: "Compression", probability: 0.20, description: "Risk of reduced expectancy and slower continuation." },
        { name: "Range", probability: 0.14, description: "Opportunity set narrows and ranking becomes more selective." },
        { name: "Stress", probability: 0.08, description: "Would typically tighten eligibility and capital posture." },
      ],
    },
    signals: {
      summary: "Signal intelligence is assembled from the latest available event and trade artifacts.",
      candidates: signals,
    },
    risk: {
      summary: "Risk radar here is inferred from repaired state artifacts until the websocket control plane is added.",
      stress: Math.min(100, Math.round(maxDrawdown * 3 + openPositions * 4)),
      slippage: Math.min(100, 24 + openPositions * 8),
      exposure: Math.min(100, Math.round((openPositions / 5) * 100)),
      guardrails,
    },
    performance,
    market,
    audit: {
      summary: "Audit rows come from persisted ledgers and event exports.",
      trades,
      events: buildEvents(events),
    },
  }));
}
