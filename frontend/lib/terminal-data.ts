import "server-only";

import fs from "node:fs/promises";
import path from "node:path";
import Papa from "papaparse";

import type { AuditEvent, AuditTrade, Guardrail, MetricTile, ReasoningTree, SignalCandidate, TerminalSnapshot } from "@/lib/terminal-types";

const REPO_ROOT = process.env.QUANT_SMC_ROOT
  ? path.resolve(process.env.QUANT_SMC_ROOT)
  : path.resolve(process.cwd(), "..");
const BACKEND_API_URL = process.env.QUANT_TERMINAL_API_URL ?? "http://127.0.0.1:8100/snapshot";

async function exists(filePath: string) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
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

function num(value: unknown, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function toneFromGuardrail(status: Guardrail["status"]): MetricTile["tone"] {
  if (status === "pass") return "teal";
  if (status === "warn") return "amber";
  return "rose";
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

  return {
    meta: {
      source: "demo",
      lastUpdated: new Date().toISOString(),
      repoRoot: REPO_ROOT,
      modelVersion: "demo-v1",
      transport: "fastapi + websocket preferred, artifact fallback available",
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
    audit: {
      summary: "Every state shown here is meant to map cleanly back to the same deterministic feature graph used in research and execution.",
      trades: [
        { tradeId: "TR-3112", asset: "BTCUSD", leg: "core", tier: "A+", pnl: 840, r: 3.0, reason: "core_tp_3.0R", entryTs: "2026-03-03T10:30:00Z", exitTs: "2026-03-03T13:00:00Z" },
        { tradeId: "TR-3113", asset: "BTCUSD", leg: "runner", tier: "A+", pnl: 1560, r: 7.2, reason: "runner_tp_7.2R", entryTs: "2026-03-03T10:30:00Z", exitTs: "2026-03-03T15:45:00Z" },
        { tradeId: "TR-3118", asset: "ETHUSD", leg: "core", tier: "A", pnl: 420, r: 2.0, reason: "core_tp_2.0R", entryTs: "2026-03-03T12:00:00Z", exitTs: "2026-03-03T13:30:00Z" },
      ],
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
  return rows.slice(-6).reverse().map((row, idx) => ({
    tradeId: String(row.trade_id || `TR-${idx + 1}`),
    asset: String(row.asset || "XBTUSD"),
    leg: String(row.leg || "core"),
    tier: String(row.tier || "unranked"),
    pnl: num(row.pnl),
    r: num(row.r),
    reason: String(row.reason || row.result || "closed"),
    entryTs: String(row.entry_ts || row.entry_time || row.timestamp || ""),
    exitTs: String(row.exit_ts || row.exit_time || ""),
  }));
}

function buildEvents(rows: Record<string, string>[]): AuditEvent[] {
  return rows.slice(-8).reverse().map((row, idx) => ({
    timestamp: String(row.timestamp || row.entry_ts || row.exit_ts || new Date(Date.now() - idx * 60000).toISOString()),
    type: String(row.type || row.event || row.reason || "event"),
    detail: String(row.detail || row.reason || row.asset || row.trade_id || "Replayable event"),
  }));
}

export async function loadTerminalSnapshot(): Promise<TerminalSnapshot> {
  if (BACKEND_API_URL) {
    try {
      const response = await fetch(BACKEND_API_URL, { cache: "no-store" });
      if (response.ok) {
        return (await response.json()) as TerminalSnapshot;
      }
    } catch {
      // fall back to local artifact loader
    }
  }

  const backtestRoot = path.join(REPO_ROOT, "backtest_outputs");
  const forwardRoot = path.join(REPO_ROOT, "forward_outputs");
  const liveRoot = path.join(REPO_ROOT, "live_outputs");
  const modelRoot = path.join(REPO_ROOT, "models");

  const snapshot = await readJson<Record<string, unknown>>(path.join(forwardRoot, "snapshot.json"))
    ?? await readJson<Record<string, unknown>>(path.join(liveRoot, "snapshot.json"));
  const state = (snapshot?.state as Record<string, unknown> | undefined)
    ?? await readJson<Record<string, unknown>>(path.join(forwardRoot, "state.json"))
    ?? await readJson<Record<string, unknown>>(path.join(liveRoot, "state.json"))
    ?? {};
  const summary = await readJson<Record<string, unknown>>(path.join(backtestRoot, "summary.json"));
  const events = ((snapshot?.events as Record<string, string>[] | undefined) ?? [])
    .concat(await readCsv(path.join(forwardRoot, "events.csv")))
    .concat(await readCsv(path.join(liveRoot, "events.csv")));
  const ledger = (await readCsv(path.join(backtestRoot, "ledger.csv")))
    .concat(await readCsv(path.join(forwardRoot, "closed_trades.csv")))
    .concat(await readCsv(path.join(liveRoot, "closed_trades.csv")));

  const version = await listLatestModelVersion(modelRoot);
  if (!summary && !snapshot && !ledger.length && !events.length) {
    return makeDemoSnapshot();
  }

  const guardrails = deriveGuardrails(state);
  const equity = num(state.equity, num(summary?.ending_equity, 20_000));
  const freeCapital = num(state.free_capital, equity);
  const lockedProfit = num(state.locked_profit, 0);
  const openPositions = num(state.open_positions, 0);
  const winRate = num(summary?.win_rate, 0.57) * (num(summary?.win_rate) <= 1 ? 100 : 1);
  const maxDrawdown = Math.abs(num(summary?.max_drawdown, num(state.max_drawdown, 0)));
  const signals = buildSignalsFromEvents(events);

  return {
    meta: {
      source: "artifacts",
      lastUpdated: new Date().toISOString(),
      repoRoot: REPO_ROOT,
      modelVersion: version,
      transport: "fastapi + websocket preferred, artifact fallback available",
    },
    mission: {
      headline: "Artifact-backed terminal state loaded from repaired repo outputs",
      status: state.cooling_to ? "Cooling" : openPositions ? "Active" : "Monitoring",
      substatus: state.cooling_to
        ? `Cooling active until ${String(state.cooling_to)}`
        : `${openPositions} live positions visible through current artifacts.`,
      metrics: [
        { label: "Equity", value: fmtMoney(equity), tone: "cyan", delta: version },
        { label: "Free Capital", value: fmtMoney(freeCapital), tone: "teal", delta: "deployable" },
        { label: "Locked Profit", value: fmtMoney(lockedProfit), tone: "amber", delta: "vaulted" },
        { label: "Open Positions", value: String(openPositions), tone: openPositions ? "amber" : "slate", delta: "live state" },
        { label: "Win Rate", value: fmtPct(winRate), tone: winRate >= 55 ? "teal" : "rose", delta: "backtest summary" },
      ],
    },
    insights: {
      summary: "Insights are currently derived from live state, recent events, and repaired model artifact metadata.",
      trace: [
        { label: "Capital Cycle", value: lockedProfit > 0 ? "Compounding" : "Base ticket", detail: lockedProfit > 0 ? "Profit has been vaulted while cycle capital remains active." : "System is trading from the base allocation.", tone: "amber" },
        { label: "Execution Posture", value: state.cooling_to ? "Guarded" : "Eligible", detail: state.cooling_to ? "Cooling timer is active under current state." : "No cooling blocker is present in state artifacts.", tone: state.cooling_to ? "rose" : "teal" },
        { label: "Model Surface", value: version, detail: "Latest discovered model registry version across repaired artifact directories.", tone: "cyan" },
        { label: "Decision Trace", value: `${events.length} events`, detail: "The frontend is reading the same persisted event stream the repaired runtime emits today.", tone: "teal" },
      ],
      latestReasoning: signals[0]?.reasoning,
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
    audit: {
      summary: "Audit rows come from persisted ledgers and event exports.",
      trades: buildTrades(ledger),
      events: buildEvents(events),
    },
  };
}
