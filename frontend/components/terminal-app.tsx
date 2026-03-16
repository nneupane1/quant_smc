"use client";

import Image from "next/image";
import { startTransition, useDeferredValue, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  AudioWaveform,
  BarChart3,
  BrainCircuit,
  CandlestickChart,
  CheckCircle2,
  CircleX,
  DatabaseZap,
  Gauge,
  Maximize2,
  Minimize2,
  Orbit,
  Pause,
  Play,
  Radar,
  RotateCcw,
  ShieldCheck,
} from "lucide-react";

import type { AuditEvent, AuditTrade, Guardrail, InsightNode, MetricTile, ReasoningTree, SignalCandidate, TerminalMode, TerminalSnapshot } from "@/lib/terminal-types";
import { MarketCanvas } from "@/components/market-canvas";

const DOMAINS = [
  { id: "mission", label: "Mission Control", caption: "Execution desk, cycle capital, open risk", icon: Activity },
  { id: "market", label: "Market Canvas", caption: "TradingView-style candles + quant overlays", icon: CandlestickChart },
  { id: "performance", label: "Performance Intel", caption: "PnL, RR, win-rate and trade ledger", icon: BarChart3 },
  { id: "insights", label: "Insights", caption: "Causal trace, structure, eligibility", icon: BrainCircuit },
  { id: "regime", label: "Regime Briefings", caption: "12h state, persistence, transition risk", icon: Orbit },
  { id: "signals", label: "Signal Intelligence", caption: "Ranked candidates, confluence, coherence", icon: AudioWaveform },
  { id: "confluence", label: "Confluence Studio", caption: "Rule vs ML confluence, stack alignment, decision quality", icon: Gauge },
  { id: "risk", label: "Risk Radar", caption: "Stress, slippage, exposure, gates", icon: Radar },
  { id: "audit", label: "Research & Audit", caption: "Traceability, trades, replayable events", icon: DatabaseZap },
] as const;

type DomainId = (typeof DOMAINS)[number]["id"];

const MODE_OPTIONS: Array<{ id: TerminalMode; label: string; caption: string }> = [
  { id: "auto", label: "Auto", caption: "smart source" },
  { id: "backtest", label: "Backtest", caption: "historical" },
  { id: "forward", label: "Forward Test", caption: "paper runtime" },
  { id: "live", label: "Live", caption: "production runtime" },
];

const toneClasses: Record<MetricTile["tone"], string> = {
  cyan: "text-cyan border-cyan/20 bg-cyan/10",
  teal: "text-teal border-teal/20 bg-teal/10",
  amber: "text-amber border-amber/20 bg-amber/10",
  rose: "text-rose border-rose/20 bg-rose/10",
  slate: "text-slate-200 border-white/10 bg-white/5",
};

function normalizeSnapshot(snapshot: TerminalSnapshot): TerminalSnapshot {
  const normalizedMeta = {
    ...snapshot.meta,
    viewModeRequested: snapshot.meta?.viewModeRequested ?? "auto",
    viewModeEffective: snapshot.meta?.viewModeEffective ?? "auto",
  };
  const normalizedMarket = snapshot.market
    ? {
        ...snapshot.market,
        zones: snapshot.market.zones ?? [],
        timeframes: snapshot.market.timeframes ?? {
          m15: snapshot.market.candles ?? [],
          h1: [],
          h6: [],
          h12: [],
        },
      }
    : {
        symbol: snapshot.signals.candidates[0]?.asset ?? "BTCUSD",
        timeframe: "15m",
        summary: "Market canvas payload not provided by backend yet.",
        candles: [],
        markers: [],
        zones: [],
        timeframes: { m15: [], h1: [], h6: [], h12: [] },
        stats: [],
        activeTrades: snapshot.audit?.trades?.slice(0, 8) ?? [],
      };
  if (snapshot.performance && snapshot.market) return { ...snapshot, meta: normalizedMeta, market: normalizedMarket };
  return {
    ...snapshot,
    meta: normalizedMeta,
    performance: {
      summary: snapshot.performance?.summary ?? "Performance payload not provided by backend yet.",
      kpis: [],
      periods: [],
      byAsset: [],
      byTier: [],
      tradeTable: snapshot.audit?.trades ?? [],
    },
    market: normalizedMarket,
  };
}

function getObject(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function pickNumber(...values: unknown[]): number | null {
  for (const value of values) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function pickString(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function normalizeTierLabel(tier: string): "Aplus" | "A" | "B" | "other" {
  const raw = tier.trim().toLowerCase();
  if (raw === "a+" || raw === "aplus" || raw === "a_plus") return "Aplus";
  if (raw === "a") return "A";
  if (raw === "b") return "B";
  return "other";
}

function buildAlertOperatorGuide({
  tier,
  confluence,
  evr,
  hazard,
  flow1h,
}: {
  tier: string;
  confluence: number;
  evr: number;
  hazard: number;
  flow1h: number;
}): {
  label: string;
  tone: MetricTile["tone"];
  summary: string;
  action: string;
  checks: Array<{ label: string; detail: string; tone: MetricTile["tone"]; status: string }>;
} {
  const tierKey = normalizeTierLabel(tier);
  const tierSpec =
    tierKey === "Aplus"
      ? { minConfluence: 0.75, minEvr: 1.5, maxHazard: 0.35 }
      : tierKey === "A"
        ? { minConfluence: 0.6, minEvr: 1.2, maxHazard: 0.45 }
        : tierKey === "B"
          ? { minConfluence: 0.45, minEvr: 0.8, maxHazard: 0.6 }
          : { minConfluence: 0.6, minEvr: 1.2, maxHazard: 0.45 };
  const flowFloor = 0.55;
  const confluenceOk = confluence >= tierSpec.minConfluence;
  const evrOk = evr >= tierSpec.minEvr;
  const hazardOk = hazard <= tierSpec.maxHazard;
  const flowOk = flow1h >= flowFloor;

  let label = "Pass";
  let tone: MetricTile["tone"] = "rose";
  let summary = "This alert is outside the normal acceptance posture. Unless there is a separate external reason to keep it visible, stand down.";
  let action = "Manual mode: pass. Auto mode: do not force an override in favor of the trade.";

  if ((tierKey === "Aplus" || tierKey === "A") && confluenceOk && evrOk && hazardOk && flowOk) {
    label = "Green Light";
    tone = "teal";
    summary = "This alert is already pre-qualified by the system and sits inside the normal acceptance posture for execution.";
    action = "Manual mode: confirm unless there is an external veto. Auto mode: allow the bot to execute.";
  } else if (confluenceOk && evrOk && hazardOk) {
    label = "Caution";
    tone = "amber";
    summary = "This alert passed the stack, but the posture is conditional rather than clean. Treat it as a smaller or more supervised process trade.";
    action = "Manual mode: confirm only if you accept a lower-quality setup. Auto mode: acceptable if policy allows B-tier or conditional entries.";
  }

  return {
    label,
    tone,
    summary,
    action,
    checks: [
      { label: "Alert Status", detail: "Triggered alerts are already pre-qualified by the modeled decision pipeline.", tone: "cyan", status: "qualified" },
      { label: "Tier", detail: `${tier} using floors conf ${tierSpec.minConfluence.toFixed(2)}, EVR ${tierSpec.minEvr.toFixed(2)}, hazard ${tierSpec.maxHazard.toFixed(2)}.`, tone: tierKey === "Aplus" || tierKey === "A" ? "teal" : tierKey === "B" ? "amber" : "rose", status: tier },
      { label: "Confluence", detail: `${confluence.toFixed(2)} vs required ${tierSpec.minConfluence.toFixed(2)}.`, tone: confluenceOk ? "teal" : "rose", status: confluenceOk ? "pass" : "fail" },
      { label: "EVR", detail: `${evr.toFixed(2)} vs required ${tierSpec.minEvr.toFixed(2)}.`, tone: evrOk ? "teal" : "rose", status: evrOk ? "pass" : "fail" },
      { label: "Hazard", detail: `${hazard.toFixed(2)} vs ceiling ${tierSpec.maxHazard.toFixed(2)}. Lower is better.`, tone: hazardOk ? "teal" : "rose", status: hazardOk ? "safe" : "high" },
      { label: "Flow 1h", detail: `${flow1h.toFixed(2)} vs guide floor ${flowFloor.toFixed(2)}.`, tone: flowOk ? "teal" : "amber", status: flowOk ? "fresh" : "thin" },
    ],
  };
}

export function TerminalApp({
  initialSnapshot,
  initialMode = "auto",
}: {
  initialSnapshot: TerminalSnapshot;
  initialMode?: TerminalMode;
}) {
  const [activeDomain, setActiveDomain] = useState<DomainId>("mission");
  const [hoveredDomain, setHoveredDomain] = useState<DomainId | null>(null);
  const [snapshot, setSnapshot] = useState(normalizeSnapshot(initialSnapshot));
  const [viewMode, setViewMode] = useState<TerminalMode>(initialMode);
  const [wsConnected, setWsConnected] = useState(false);
  const [selectedSignalId, setSelectedSignalId] = useState<string | null>(initialSnapshot.signals.candidates[0]?.id ?? null);
  const deferredSignals = useDeferredValue(snapshot.signals.candidates).slice(0, 5);
  const wsUrl = useMemo(() => {
    if (typeof window === "undefined") {
      return process.env.NEXT_PUBLIC_TERMINAL_WS_URL ?? null;
    }
    if (process.env.NEXT_PUBLIC_TERMINAL_WS_URL) {
      return process.env.NEXT_PUBLIC_TERMINAL_WS_URL;
    }
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    return `${protocol}://${window.location.hostname}:8100/ws/terminal`;
  }, []);

  useEffect(() => {
    let cancelled = false;
    const refreshSnapshot = async () => {
      try {
        const res = await fetch(`/api/terminal?mode=${viewMode}`, { cache: "no-store" });
        if (!res.ok || cancelled) return;
        const next = (await res.json()) as TerminalSnapshot;
        if (!cancelled) {
          startTransition(() => setSnapshot(normalizeSnapshot(next)));
        }
      } catch {
        // keep last successful state
      }
    };

    if (!wsConnected || (viewMode !== "auto" && viewMode !== "live")) {
      refreshSnapshot();
    }
    const timer = window.setInterval(() => {
      if (!wsConnected || (viewMode !== "auto" && viewMode !== "live")) {
        refreshSnapshot();
      }
    }, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [viewMode, wsConnected]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    url.searchParams.set("mode", viewMode);
    window.history.replaceState({}, "", url);
  }, [viewMode]);

  useEffect(() => {
    const nextId = snapshot.signals.candidates[0]?.id ?? null;
    if (!nextId) {
      if (selectedSignalId !== null) {
        setSelectedSignalId(null);
      }
      return;
    }
    if (!selectedSignalId || !snapshot.signals.candidates.some((candidate) => candidate.id === selectedSignalId)) {
      setSelectedSignalId(nextId);
    }
  }, [snapshot.signals.candidates, selectedSignalId]);

  useEffect(() => {
    if (!wsUrl) {
      return;
    }

    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let closedByCleanup = false;

    const connect = () => {
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        setWsConnected(true);
      };

      socket.onmessage = (message) => {
        if (viewMode !== "auto" && viewMode !== "live") {
          return;
        }
        try {
          const payload = JSON.parse(message.data) as {
            type?: string;
            data?: TerminalSnapshot | { snapshot?: TerminalSnapshot; events?: AuditEvent[] } | AuditEvent;
          };
          if (payload.type === "terminal_snapshot" && payload.data) {
            startTransition(() => setSnapshot(normalizeSnapshot(payload.data as TerminalSnapshot)));
          } else if (payload.type === "bootstrap" && payload.data && "snapshot" in payload.data) {
            startTransition(() => setSnapshot(normalizeSnapshot((payload.data as { snapshot: TerminalSnapshot }).snapshot)));
          } else if (payload.type === "terminal_event" && payload.data) {
            const rawEvent = payload.data as Record<string, unknown>;
            const liveEvent: AuditEvent = {
              timestamp: String(rawEvent.timestamp || ""),
              type: String(rawEvent.event_type || rawEvent.type || "event"),
              detail: String(
                (rawEvent.payload as Record<string, unknown> | undefined)?.reason
                  || (rawEvent.payload as Record<string, unknown> | undefined)?.detail
                  || rawEvent.trade_id
                  || rawEvent.event_type
                  || "event"
              ),
            };
            startTransition(() =>
              setSnapshot((prev) => ({
                ...prev,
                audit: {
                  ...prev.audit,
                  events: [liveEvent, ...prev.audit.events].slice(0, 12),
                },
              }))
            );
          }
        } catch {
          // ignore malformed messages
        }
      };

      socket.onerror = () => {
        setWsConnected(false);
      };

      socket.onclose = () => {
        setWsConnected(false);
        if (!closedByCleanup) {
          reconnectTimer = window.setTimeout(connect, 4000);
        }
      };
    };

    connect();

    return () => {
      closedByCleanup = true;
      setWsConnected(false);
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
      }
      socket?.close();
    };
  }, [viewMode, wsUrl]);

  const hoverCaption = hoveredDomain
    ? DOMAINS.find((item) => item.id === hoveredDomain)?.caption
    : DOMAINS.find((item) => item.id === activeDomain)?.caption;
  const selectedCandidate =
    snapshot.signals.candidates.find((candidate) => candidate.id === selectedSignalId)
    ?? snapshot.signals.candidates[0];
  const selectedReasoning = selectedCandidate?.reasoning ?? snapshot.insights.latestReasoning;

  return (
    <main className="terminal-shell">
      <div className="pointer-events-none absolute inset-0 bg-grid bg-[size:48px_48px] opacity-[0.08]" />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan/60 to-transparent" />

      <div className="mx-auto flex min-h-screen max-w-[1700px] gap-6 px-4 py-4 md:px-6 lg:px-8">
        <aside className="hidden w-[290px] shrink-0 flex-col lg:flex">
          <div className="glass-panel p-5">
            <div className="section-kicker">Quant SMC</div>
            <div className="mt-2 font-[var(--font-display)] text-2xl font-semibold tracking-tight text-white">
              Intelligence Terminal
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-300/75">
              Bloomberg-grade layout, with FastAPI + WebSocket terminal streaming and artifact fallback when the backend is absent.
            </p>

            <div className="mt-6 space-y-3">
              {DOMAINS.map((domain) => {
                const Icon = domain.icon;
                const active = activeDomain === domain.id;
                return (
                  <button
                    key={domain.id}
                    className={`group nav-chip w-full ${active ? "nav-chip-active" : ""}`}
                    onClick={() => setActiveDomain(domain.id)}
                    onMouseEnter={() => setHoveredDomain(domain.id)}
                    onMouseLeave={() => setHoveredDomain(null)}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`mt-0.5 rounded-2xl border p-2 ${active ? "border-cyan/40 bg-cyan/10 text-cyan" : "border-white/10 bg-white/5 text-slate-300"}`}>
                        <Icon className="h-4 w-4" />
                      </div>
                      <div>
                        <div className="font-medium text-white">{domain.label}</div>
                        <div className="mt-1 text-xs leading-5 text-slate-400">{domain.caption}</div>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="glass-panel mt-4 overflow-hidden p-5">
            <div className="section-kicker">Hover Lens</div>
            <div className="mt-3 text-sm text-slate-300">{hoverCaption}</div>
            <div className="mt-4 rounded-2xl border border-amber/15 bg-amber/10 p-4 text-sm text-amber-100/90 shadow-amber">
              Transport: {wsConnected ? "websocket live" : snapshot.meta.transport}
            </div>
          </div>
        </aside>

        <div className="min-w-0 flex-1">
          <TopBar snapshot={snapshot} wsConnected={wsConnected} />
          <ModeDock
            selectedMode={viewMode}
            effectiveMode={snapshot.meta.viewModeEffective ?? viewMode}
            onChange={setViewMode}
          />
          <div className={`mt-5 grid gap-5 ${activeDomain === "market" ? "xl:grid-cols-1" : "xl:grid-cols-[1.5fr_0.95fr]"}`}>
            <AnimatePresence mode="wait">
              <motion.section
                key={activeDomain}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.24, ease: "easeOut" }}
                className={activeDomain === "market" ? "min-w-0 xl:col-span-1" : "min-w-0"}
              >
                {activeDomain === "mission" && <MissionPanel snapshot={snapshot} />}
                {activeDomain === "market" && <MarketPanel snapshot={snapshot} />}
                {activeDomain === "performance" && <PerformancePanel snapshot={snapshot} />}
                {activeDomain === "insights" && <InsightsPanel summary={snapshot.insights.summary} trace={snapshot.insights.trace} reasoning={selectedReasoning} />}
                {activeDomain === "regime" && <RegimePanel snapshot={snapshot} />}
                {activeDomain === "signals" && (
                  <SignalsPanel
                    summary={snapshot.signals.summary}
                    candidates={deferredSignals}
                    selectedSignalId={selectedSignalId}
                    onSelectSignal={setSelectedSignalId}
                    reasoning={selectedReasoning}
                  />
                )}
                {activeDomain === "confluence" && (
                  <ConfluencePanel
                    snapshot={snapshot}
                    candidates={snapshot.signals.candidates}
                    selectedSignalId={selectedSignalId}
                    onSelectSignal={setSelectedSignalId}
                    reasoning={selectedReasoning}
                  />
                )}
                {activeDomain === "risk" && <RiskPanel snapshot={snapshot} />}
                {activeDomain === "audit" && <AuditPanel snapshot={snapshot} />}
              </motion.section>
            </AnimatePresence>

            {activeDomain !== "market" ? <RightRail snapshot={snapshot} /> : null}
          </div>
        </div>
      </div>
    </main>
  );
}

function TopBar({ snapshot, wsConnected }: { snapshot: TerminalSnapshot; wsConnected: boolean }) {
  return (
    <div className="glass-panel overflow-hidden px-5 py-5">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan/70 to-transparent" />
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div className="flex items-center gap-7">
          <div className="relative hidden h-[176px] w-[336px] shrink-0 overflow-hidden rounded-[34px] border border-cyan/30 bg-[#030c15] shadow-bloom lg:block">
            <div className="absolute inset-[5px] overflow-hidden rounded-[30px] bg-[#020912]">
              <Image
                src="/bull_bear.png"
                alt="Quant SMC Bull Bear"
                fill
                priority
                className="object-cover object-center scale-[1.34] [clip-path:inset(0_16%_0_16%)]"
              />
              <div className="pointer-events-none absolute inset-y-0 left-0 w-16 bg-gradient-to-r from-[#020912] to-transparent" />
              <div className="pointer-events-none absolute inset-y-0 right-0 w-16 bg-gradient-to-l from-[#020912] to-transparent" />
              <div className="pointer-events-none absolute inset-0 bg-gradient-to-tr from-cyan/18 via-transparent to-amber/20" />
            </div>
            <div className="pointer-events-none absolute inset-0 rounded-[34px] ring-1 ring-inset ring-cyan/18" />
          </div>
          <div>
            <div className="section-kicker">Live Control Room</div>
            <div className="mt-2 font-[var(--font-display)] text-3xl font-semibold tracking-tight text-cyan md:text-4xl xl:whitespace-nowrap">
              Quant Execution Cockpit
            </div>
            <p className="mt-2 max-w-3xl text-xs uppercase tracking-[0.18em] text-amber/80">{snapshot.mission.headline}</p>
            <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-300/75">{snapshot.mission.substatus}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-3 xl:max-w-[520px] xl:justify-end">
          <StatusPill label={snapshot.meta.source === "artifacts" ? "Artifact parity live" : "Demo parity preview"} tone={snapshot.meta.source === "artifacts" ? "teal" : "amber"} />
          <StatusPill label={wsConnected ? "WS linked" : "WS standby"} tone={wsConnected ? "teal" : "amber"} />
          <StatusPill label={`Models ${snapshot.meta.modelVersion}`} tone="cyan" />
          <StatusPill label={snapshot.mission.status} tone={snapshot.mission.status === "Cooling" ? "rose" : "teal"} />
        </div>
      </div>
    </div>
  );
}

function ModeDock({
  selectedMode,
  effectiveMode,
  onChange,
}: {
  selectedMode: TerminalMode;
  effectiveMode: TerminalMode;
  onChange: (mode: TerminalMode) => void;
}) {
  return (
    <div className="glass-panel mt-4 overflow-hidden px-5 py-5">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div>
          <div className="section-kicker">Environment Mode</div>
          <div className="mt-2 text-sm leading-6 text-slate-300/75">
            Switch the entire terminal between historical, forward-test, and live runtime views without changing the shell.
          </div>
        </div>
        <div className="rounded-full border border-amber/20 bg-amber/10 px-4 py-2 text-xs uppercase tracking-[0.2em] text-amber">
          effective {effectiveMode}
        </div>
      </div>
      <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {MODE_OPTIONS.map((mode) => {
          const active = selectedMode === mode.id;
          return (
            <button
              key={mode.id}
              onClick={() => onChange(mode.id)}
              className={`rounded-[22px] border px-4 py-4 text-left transition ${
                active
                  ? "border-amber/40 bg-amber/12 text-amber shadow-amber"
                  : "border-white/10 bg-white/[0.03] text-slate-200 hover:border-cyan/30 hover:bg-cyan/5"
              }`}
            >
              <div className="font-medium">{mode.label}</div>
              <div className="mt-1 text-[0.68rem] uppercase tracking-[0.18em] text-slate-400">{mode.caption}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function MissionPanel({ snapshot }: { snapshot: TerminalSnapshot }) {
  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {snapshot.mission.metrics.map((metric) => (
          <MetricCard key={metric.label} metric={metric} />
        ))}
      </div>
      <div className="glass-panel p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="section-kicker">Execution Desk</div>
            <h2 className="mt-2 text-xl font-semibold text-white">Deterministic cycle capital surface</h2>
          </div>
          <div className="rounded-full border border-cyan/25 bg-cyan/10 px-4 py-2 text-xs uppercase tracking-[0.24em] text-cyan">
            {snapshot.meta.source}
          </div>
        </div>
        <div className="mt-5 grid gap-4 md:grid-cols-3">
          <NarrativePanel
            title="Compounding Loop"
            text="Cycle capital compounds while deployable equity grows. Profit vaulting and cooling remain explicit state transitions, not invisible heuristics."
          />
          <NarrativePanel
            title="Research Parity"
            text="The same repaired feature, label, model, and execution contracts are the source of truth for backtest, forward, live, and this terminal."
          />
          <NarrativePanel
            title="Guarded Aggression"
            text="Profit ladder, hazard trailing, and cooling logic are visible as operator state, not buried inside opaque back-end reactions."
          />
        </div>
      </div>
    </div>
  );
}

function MarketPanel({ snapshot }: { snapshot: TerminalSnapshot }) {
  const market = snapshot.market;
  const panelRef = useRef<HTMLDivElement | null>(null);
  const [activeTf, setActiveTf] = useState<"15m" | "1h" | "6h" | "12h">("15m");
  const [showStats, setShowStats] = useState(true);
  const [showTrades, setShowTrades] = useState(true);
  const [showReasoning, setShowReasoning] = useState(true);
  const [showOb, setShowOb] = useState(true);
  const [showFvg, setShowFvg] = useState(true);
  const [showLiquidity, setShowLiquidity] = useState(true);
  const [isCinematic, setIsCinematic] = useState(false);
  const [replayIndex, setReplayIndex] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [cursorTime, setCursorTime] = useState<number | null>(null);

  const tfMap = useMemo(() => ({
    "15m": market.timeframes?.m15?.length ? market.timeframes.m15 : market.candles,
    "1h": market.timeframes?.h1 ?? [],
    "6h": market.timeframes?.h6 ?? [],
    "12h": market.timeframes?.h12 ?? [],
  }), [market.candles, market.timeframes]);
  const activeCandles = tfMap[activeTf].length ? tfMap[activeTf] : market.candles;
  const maxReplayIndex = Math.max(activeCandles.length - 1, 0);
  const focusTrades = (market.activeTrades?.length ? market.activeTrades : snapshot.performance.tradeTable).slice(0, 12);

  useEffect(() => {
    setReplayIndex(null);
    setIsPlaying(false);
    setCursorTime(null);
  }, [activeTf]);

  useEffect(() => {
    if (replayIndex === null) return;
    if (replayIndex > maxReplayIndex) {
      setReplayIndex(maxReplayIndex);
    }
  }, [maxReplayIndex, replayIndex]);

  useEffect(() => {
    const onFullscreen = () => {
      setIsCinematic(document.fullscreenElement === panelRef.current);
    };
    document.addEventListener("fullscreenchange", onFullscreen);
    return () => document.removeEventListener("fullscreenchange", onFullscreen);
  }, []);

  useEffect(() => {
    if (!isPlaying || !activeCandles.length) return;
    const timer = window.setInterval(() => {
      setReplayIndex((current) => {
        const start = current ?? Math.max(0, maxReplayIndex - 220);
        const next = Math.min(maxReplayIndex, start + 1);
        if (next >= maxReplayIndex) {
          setIsPlaying(false);
        }
        return next;
      });
    }, 320);
    return () => window.clearInterval(timer);
  }, [activeCandles.length, isPlaying, maxReplayIndex]);

  const replayTime = replayIndex !== null ? activeCandles[Math.min(replayIndex, maxReplayIndex)]?.time ?? null : null;
  const focusTime = cursorTime ?? replayTime ?? activeCandles[activeCandles.length - 1]?.time ?? null;
  const visibleCandles = replayIndex === null ? activeCandles : activeCandles.slice(0, replayIndex + 1);
  const visibleZones = useMemo(
    () =>
      (market.zones ?? []).filter((zone) => {
        if (zone.kind === "ob" && !showOb) return false;
        if (zone.kind === "fvg" && !showFvg) return false;
        if (zone.kind === "liquidity" && !showLiquidity) return false;
        if (replayTime === null) return true;
        return zone.start <= replayTime;
      }),
    [market.zones, replayTime, showFvg, showLiquidity, showOb],
  );
  const replayPct = replayIndex === null || maxReplayIndex === 0
    ? 100
    : Math.min(100, ((replayIndex + 1) / (maxReplayIndex + 1)) * 100);

  const toggleCinematic = async () => {
    const host = panelRef.current;
    if (!host) return;
    if (document.fullscreenElement === host) {
      await document.exitFullscreen();
      setIsCinematic(false);
      return;
    }
    await host.requestFullscreen();
    setIsCinematic(true);
  };

  return (
    <div
      ref={panelRef}
      className={`space-y-5 ${isCinematic ? "fixed inset-0 z-[120] overflow-y-auto bg-[#040a13]/95 p-5 md:p-6" : ""}`}
    >
      <div className="glass-panel p-5">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
          <div>
            <div className="section-kicker">Unified Market Canvas</div>
            <h2 className="mt-2 text-xl font-semibold text-white">
              {market.symbol} • {activeTf}
            </h2>
            <p className="mt-3 text-sm leading-6 text-slate-300/80">
              {market.summary}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="rounded-full border border-cyan/30 bg-cyan/10 px-4 py-2 text-xs uppercase tracking-[0.2em] text-cyan">
              {visibleCandles.length} bars • {visibleZones.length} overlays
            </div>
            <button
              className="rounded-full border border-white/15 bg-white/[0.04] px-3 py-2 text-xs uppercase tracking-[0.16em] text-slate-200 transition hover:border-cyan/40 hover:text-cyan"
              onClick={toggleCinematic}
            >
              {isCinematic ? <Minimize2 className="inline h-4 w-4" /> : <Maximize2 className="inline h-4 w-4" />} {isCinematic ? "Exit cinematic" : "Cinematic"}
            </button>
          </div>
        </div>
      </div>

      <div className="glass-panel p-4">
        <div className="flex flex-wrap items-center gap-2">
          {(["12h", "6h", "1h", "15m"] as const).map((tf) => (
            <button
              key={tf}
              onClick={() => setActiveTf(tf)}
              className={`rounded-full border px-3 py-2 text-xs uppercase tracking-[0.16em] transition ${
                activeTf === tf
                  ? "border-cyan/45 bg-cyan/15 text-cyan shadow-bloom"
                  : "border-white/15 bg-white/[0.03] text-slate-300 hover:border-cyan/35 hover:text-cyan"
              }`}
            >
              {tf}
            </button>
          ))}
          <div className="mx-2 h-6 w-px bg-white/10" />
          <button
            onClick={() => setShowOb((current) => !current)}
            className={`rounded-full border px-3 py-2 text-xs uppercase tracking-[0.16em] transition ${showOb ? "border-teal/45 bg-teal/15 text-teal" : "border-white/15 bg-white/[0.03] text-slate-300"}`}
          >
            OB
          </button>
          <button
            onClick={() => setShowFvg((current) => !current)}
            className={`rounded-full border px-3 py-2 text-xs uppercase tracking-[0.16em] transition ${showFvg ? "border-amber/45 bg-amber/15 text-amber" : "border-white/15 bg-white/[0.03] text-slate-300"}`}
          >
            FVG
          </button>
          <button
            onClick={() => setShowLiquidity((current) => !current)}
            className={`rounded-full border px-3 py-2 text-xs uppercase tracking-[0.16em] transition ${showLiquidity ? "border-rose/45 bg-rose/15 text-rose" : "border-white/15 bg-white/[0.03] text-slate-300"}`}
          >
            Liquidity
          </button>
          <div className="mx-2 h-6 w-px bg-white/10" />
          <button
            onClick={() => {
              setReplayIndex((current) => current ?? Math.max(0, maxReplayIndex - 220));
              setIsPlaying((current) => !current);
            }}
            className="rounded-full border border-white/15 bg-white/[0.03] px-3 py-2 text-xs uppercase tracking-[0.16em] text-slate-200 transition hover:border-cyan/35 hover:text-cyan"
          >
            {isPlaying ? <Pause className="inline h-4 w-4" /> : <Play className="inline h-4 w-4" />} {isPlaying ? "Pause" : "Play"}
          </button>
          <button
            onClick={() => {
              setReplayIndex(null);
              setIsPlaying(false);
              setCursorTime(null);
            }}
            className="rounded-full border border-white/15 bg-white/[0.03] px-3 py-2 text-xs uppercase tracking-[0.16em] text-slate-200 transition hover:border-cyan/35 hover:text-cyan"
          >
            <RotateCcw className="inline h-4 w-4" /> Live
          </button>
          <div className="mx-2 h-6 w-px bg-white/10" />
          <button
            onClick={() => setShowStats((current) => !current)}
            className={`rounded-full border px-3 py-2 text-xs uppercase tracking-[0.16em] transition ${showStats ? "border-cyan/45 bg-cyan/15 text-cyan" : "border-white/15 bg-white/[0.03] text-slate-300"}`}
          >
            KPI Panel
          </button>
          <button
            onClick={() => setShowTrades((current) => !current)}
            className={`rounded-full border px-3 py-2 text-xs uppercase tracking-[0.16em] transition ${showTrades ? "border-cyan/45 bg-cyan/15 text-cyan" : "border-white/15 bg-white/[0.03] text-slate-300"}`}
          >
            Trades
          </button>
          <button
            onClick={() => setShowReasoning((current) => !current)}
            className={`rounded-full border px-3 py-2 text-xs uppercase tracking-[0.16em] transition ${showReasoning ? "border-cyan/45 bg-cyan/15 text-cyan" : "border-white/15 bg-white/[0.03] text-slate-300"}`}
          >
            Reasoning
          </button>
        </div>
        {replayIndex !== null ? (
          <div className="mt-4 space-y-2">
            <input
              type="range"
              min={0}
              max={Math.max(0, maxReplayIndex)}
              step={1}
              value={replayIndex}
              onChange={(event) => {
                setReplayIndex(Number(event.target.value));
                setIsPlaying(false);
              }}
              className="w-full accent-cyan"
            />
            <div className="flex items-center justify-between text-xs uppercase tracking-[0.14em] text-slate-400">
              <span>Replay {replayPct.toFixed(1)}%</span>
              <span>{replayTime ? fmtUnix(replayTime) : "live edge"}</span>
            </div>
          </div>
        ) : null}
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {(["12h", "6h", "1h", "15m"] as const).map((tf) => (
          <TimeframeMiniMap
            key={tf}
            label={tf}
            candles={tfMap[tf]}
            active={activeTf === tf}
            focusTime={focusTime}
            onClick={() => setActiveTf(tf)}
          />
        ))}
      </div>

      <div className="glass-panel p-2 md:p-3">
        <MarketCanvas
          market={{ ...market, timeframe: activeTf, candles: visibleCandles, zones: visibleZones }}
          onHoverTime={setCursorTime}
          focusTime={focusTime}
        />
      </div>

      {showStats ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {market.stats.map((stat) => (
            <MetricCard key={stat.label} metric={{ label: stat.label, value: stat.value, tone: stat.tone, delta: stat.detail }} />
          ))}
        </div>
      ) : null}

      <div className={`grid gap-4 ${showTrades && showReasoning ? "xl:grid-cols-[1.2fr_1fr]" : "xl:grid-cols-1"}`}>
        {showTrades ? (
          <div className="glass-panel overflow-hidden">
            <div className="border-b border-white/10 px-5 py-4">
              <div className="section-kicker">Trade Context Table</div>
              <div className="mt-2 text-sm text-slate-300">Execution details directly aligned with the active candle canvas.</div>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead className="border-b border-white/10 bg-white/[0.03] text-xs uppercase tracking-[0.18em] text-slate-400">
                  <tr>
                    {["Trade", "Side", "Tier", "PnL", "R", "Entry", "Exit", "Hold", "Model", "Reason"].map((header) => (
                      <th key={header} className="px-3 py-3 font-medium">{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {focusTrades.map((trade) => (
                    <tr key={`${trade.tradeId}-${trade.entryTs}`} className="border-b border-white/5 transition hover:bg-cyan/5">
                      <td className="px-3 py-3 font-mono text-xs text-slate-300">{trade.tradeId}</td>
                      <td className={`px-3 py-3 ${trade.side === "short" ? "text-rose" : "text-teal"}`}>{trade.side ?? "long"}</td>
                      <td className="px-3 py-3 text-amber">{trade.tier}</td>
                      <td className={`px-3 py-3 ${trade.pnl >= 0 ? "text-teal" : "text-rose"}`}>{fmtSignedUsd(trade.pnl)}</td>
                      <td className="px-3 py-3 text-cyan">{trade.r.toFixed(2)}R</td>
                      <td className="px-3 py-3 text-slate-400">{fmtTs(trade.entryTs)}</td>
                      <td className="px-3 py-3 text-slate-400">{fmtTs(trade.exitTs || "")}</td>
                      <td className="px-3 py-3 text-slate-300">{trade.holdMinutes ? `${Math.round(trade.holdMinutes)}m` : "-"}</td>
                      <td className="px-3 py-3 text-slate-300">{trade.model || "-"}</td>
                      <td className="px-3 py-3 text-slate-400">{trade.reason}</td>
                    </tr>
                  ))}
                  {focusTrades.length === 0 ? (
                    <tr>
                      <td colSpan={10} className="px-4 py-6 text-center text-slate-400">No trade context rows yet.</td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}

        {showReasoning ? (
          <ReasoningTreePanel
            title="Latest Alert Reasoning"
            subtitle="Quant state, model outputs, and constraints that explain the latest executable intent."
            reasoning={snapshot.insights.latestReasoning}
          />
        ) : null}
      </div>
    </div>
  );
}

function PerformancePanel({ snapshot }: { snapshot: TerminalSnapshot }) {
  type Lookback = "all" | "30d" | "90d" | "180d" | "365d";
  type SortKey = "entryTs" | "exitTs" | "pnl" | "r" | "holdMinutes" | "fees" | "slippageBps";
  const allRows = snapshot.performance.tradeTable;
  const [selectedTradeId, setSelectedTradeId] = useState<string | null>(allRows[0]?.tradeId ?? null);
  const [lookback, setLookback] = useState<Lookback>("all");
  const [sideFilter, setSideFilter] = useState<"all" | "long" | "short">("all");
  const [tierFilter, setTierFilter] = useState("all");
  const [modelFilter, setModelFilter] = useState("all");
  const [sessionFilter, setSessionFilter] = useState("all");
  const [searchText, setSearchText] = useState("");
  const [sortBy, setSortBy] = useState<SortKey>("entryTs");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(0);
  const pageSize = 32;
  const verdict = snapshot.performance.verdict;

  const tierOptions = useMemo(
    () => ["all", ...Array.from(new Set(allRows.map((trade) => trade.tier).filter((value) => Boolean(value)))).sort()],
    [allRows],
  );
  const modelOptions = useMemo(
    () => ["all", ...Array.from(new Set(allRows.map((trade) => trade.model || "unknown"))).sort()],
    [allRows],
  );
  const sessionOptions = useMemo(
    () => ["all", ...Array.from(new Set(allRows.map((trade) => trade.session || "unknown"))).sort()],
    [allRows],
  );

  const filteredRows = useMemo(() => {
    const windowDays = lookback === "all" ? null : Number.parseInt(lookback.replace("d", ""), 10);
    const cutoff = windowDays ? Date.now() - windowDays * 24 * 60 * 60 * 1000 : Number.NEGATIVE_INFINITY;
    const query = searchText.trim().toLowerCase();

    return allRows.filter((trade) => {
      if (sideFilter !== "all" && (trade.side || "long") !== sideFilter) return false;
      if (tierFilter !== "all" && trade.tier !== tierFilter) return false;
      if (modelFilter !== "all" && (trade.model || "unknown") !== modelFilter) return false;
      if (sessionFilter !== "all" && (trade.session || "unknown") !== sessionFilter) return false;
      if (windowDays !== null) {
        const tradeTs = tradeTsMs(trade);
        if (!Number.isFinite(tradeTs) || tradeTs < cutoff) return false;
      }
      if (!query) return true;
      return [
        trade.tradeId,
        trade.asset,
        trade.tier,
        trade.model || "",
        trade.session || "",
        trade.regime || "",
        trade.reason,
      ]
        .join(" ")
        .toLowerCase()
        .includes(query);
    });
  }, [allRows, lookback, modelFilter, searchText, sessionFilter, sideFilter, tierFilter]);

  const sortedRows = useMemo(() => {
    const rows = [...filteredRows];
    rows.sort((a, b) => {
      const dir = sortDir === "asc" ? 1 : -1;
      if (sortBy === "entryTs" || sortBy === "exitTs") {
        const aTs = Date.parse((sortBy === "entryTs" ? a.entryTs : (a.exitTs || a.entryTs || "")) || "");
        const bTs = Date.parse((sortBy === "entryTs" ? b.entryTs : (b.exitTs || b.entryTs || "")) || "");
        if (!Number.isFinite(aTs) && !Number.isFinite(bTs)) return 0;
        if (!Number.isFinite(aTs)) return 1 * dir;
        if (!Number.isFinite(bTs)) return -1 * dir;
        return (aTs - bTs) * dir;
      }
      const aValue = Number(a[sortBy] ?? 0);
      const bValue = Number(b[sortBy] ?? 0);
      return (aValue - bValue) * dir;
    });
    return rows;
  }, [filteredRows, sortBy, sortDir]);

  const pageCount = Math.max(1, Math.ceil(sortedRows.length / pageSize));
  const pageRows = useMemo(
    () => sortedRows.slice(page * pageSize, page * pageSize + pageSize),
    [page, sortedRows],
  );

  useEffect(() => {
    setPage((current) => Math.min(current, pageCount - 1));
  }, [pageCount]);

  useEffect(() => {
    if (!sortedRows.length) {
      setSelectedTradeId(null);
      return;
    }
    if (!selectedTradeId || !sortedRows.some((trade) => trade.tradeId === selectedTradeId)) {
      setSelectedTradeId(sortedRows[0].tradeId);
    }
  }, [selectedTradeId, sortedRows]);

  const selectedTrade = sortedRows.find((trade) => trade.tradeId === selectedTradeId) ?? pageRows[0];

  const chronological = useMemo(
    () => [...sortedRows].sort((a, b) => {
      const aTs = tradeTsMs(a);
      const bTs = tradeTsMs(b);
      if (!Number.isFinite(aTs) && !Number.isFinite(bTs)) return 0;
      if (!Number.isFinite(aTs)) return 1;
      if (!Number.isFinite(bTs)) return -1;
      return aTs - bTs;
    }),
    [sortedRows],
  );

  const equitySeries = useMemo(() => {
    let running = 20_000;
    let peak = running;
    return chronological.map((trade, index) => {
      running += trade.pnl;
      peak = Math.max(peak, running);
      const drawdown = running - peak;
      return {
        label: fmtTs(trade.exitTs || trade.entryTs || ""),
        pnl: trade.pnl,
        equity: running,
        drawdown,
        trades: index + 1,
      };
    });
  }, [chronological]);

  const equityValues = useMemo(() => equitySeries.map((point) => point.equity), [equitySeries]);
  const drawdownValues = useMemo(() => equitySeries.map((point) => point.drawdown), [equitySeries]);
  const pnlSequence = useMemo(() => chronological.map((trade) => trade.pnl), [chronological]);
  const rSequence = useMemo(() => chronological.map((trade) => trade.r), [chronological]);
  const sequenceLabels = useMemo(
    () => chronological.map((trade) => fmtTs(trade.exitTs || trade.entryTs || "")),
    [chronological],
  );

  const bucketFromRows = (selector: (trade: AuditTrade) => string, limit = 8): Array<{ label: string; pnl: number; trades: number; winRate: number }> => {
    const map = new Map<string, { pnl: number; trades: number; wins: number }>();
    for (const trade of sortedRows) {
      const key = selector(trade) || "unknown";
      const slot = map.get(key) ?? { pnl: 0, trades: 0, wins: 0 };
      slot.pnl += trade.pnl;
      slot.trades += 1;
      if (trade.pnl > 0) slot.wins += 1;
      map.set(key, slot);
    }
    return [...map.entries()]
      .map(([label, slot]) => ({
        label,
        pnl: slot.pnl,
        trades: slot.trades,
        winRate: slot.trades ? (slot.wins / slot.trades) * 100 : 0,
      }))
      .sort((a, b) => b.pnl - a.pnl)
      .slice(0, limit);
  };

  const byAsset = useMemo(() => bucketFromRows((trade) => trade.asset), [sortedRows]);
  const byTier = useMemo(() => bucketFromRows((trade) => trade.tier), [sortedRows]);
  const byModel = useMemo(() => bucketFromRows((trade) => trade.model || "unknown"), [sortedRows]);
  const bySession = useMemo(() => bucketFromRows((trade) => trade.session || "unknown"), [sortedRows]);
  const byRegime = useMemo(() => bucketFromRows((trade) => trade.regime || "unknown"), [sortedRows]);
  const byHold = useMemo(() => {
    const defs = [
      { label: "<15m", min: 0, max: 15 },
      { label: "15m-1h", min: 15, max: 60 },
      { label: "1h-4h", min: 60, max: 240 },
      { label: "4h-12h", min: 240, max: 720 },
      { label: ">12h", min: 720, max: Number.POSITIVE_INFINITY },
    ];
    const map = new Map<string, { pnl: number; trades: number; wins: number }>();
    for (const def of defs) map.set(def.label, { pnl: 0, trades: 0, wins: 0 });
    for (const trade of sortedRows) {
      const holdMinutes = Math.max(0, Number(trade.holdMinutes || 0));
      const bucket = defs.find((def) => holdMinutes >= def.min && holdMinutes < def.max);
      if (!bucket) continue;
      const slot = map.get(bucket.label)!;
      slot.pnl += trade.pnl;
      slot.trades += 1;
      if (trade.pnl > 0) slot.wins += 1;
    }
    return defs.map((def) => {
      const slot = map.get(def.label)!;
      return {
        label: def.label,
        pnl: slot.pnl,
        trades: slot.trades,
        winRate: slot.trades ? (slot.wins / slot.trades) * 100 : 0,
      };
    });
  }, [sortedRows]);

  const byHour = useMemo(() => {
    const map = new Map<string, { pnl: number; trades: number; wins: number }>();
    for (let hour = 0; hour < 24; hour += 1) {
      map.set(`${String(hour).padStart(2, "0")}:00`, { pnl: 0, trades: 0, wins: 0 });
    }
    for (const trade of sortedRows) {
      const parsed = tradeTsMs(trade);
      if (!Number.isFinite(parsed)) continue;
      const hour = new Date(parsed).getUTCHours();
      const key = `${String(hour).padStart(2, "0")}:00`;
      const slot = map.get(key)!;
      slot.pnl += trade.pnl;
      slot.trades += 1;
      if (trade.pnl > 0) slot.wins += 1;
    }
    return [...map.entries()].map(([label, slot]) => ({
      label,
      pnl: slot.pnl,
      trades: slot.trades,
      winRate: slot.trades ? (slot.wins / slot.trades) * 100 : 0,
    }));
  }, [sortedRows]);

  const weekdayRows = useMemo(() => {
    const order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    const map = new Map<string, { pnl: number; trades: number; wins: number }>();
    for (const day of order) map.set(day, { pnl: 0, trades: 0, wins: 0 });
    for (const trade of sortedRows) {
      const parsed = tradeTsMs(trade);
      if (!Number.isFinite(parsed)) continue;
      const weekday = (new Date(parsed).getUTCDay() + 6) % 7;
      const key = order[weekday];
      const slot = map.get(key)!;
      slot.pnl += trade.pnl;
      slot.trades += 1;
      if (trade.pnl > 0) slot.wins += 1;
    }
    return order.map((label) => {
      const slot = map.get(label)!;
      return {
        label,
        pnl: slot.pnl,
        trades: slot.trades,
        winRate: slot.trades ? (slot.wins / slot.trades) * 100 : 0,
      };
    });
  }, [sortedRows]);

  const monthlyTimeline = useMemo(() => {
    const map = new Map<string, { pnl: number; trades: number; wins: number; rSum: number }>();
    for (const trade of chronological) {
      const parsed = tradeTsMs(trade);
      if (!Number.isFinite(parsed)) continue;
      const dt = new Date(parsed);
      const key = `${dt.getUTCFullYear()}-${String(dt.getUTCMonth() + 1).padStart(2, "0")}`;
      const slot = map.get(key) ?? { pnl: 0, trades: 0, wins: 0, rSum: 0 };
      slot.pnl += trade.pnl;
      slot.trades += 1;
      if (trade.pnl > 0) slot.wins += 1;
      slot.rSum += trade.r;
      map.set(key, slot);
    }
    return [...map.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([label, slot]) => ({
        label,
        pnl: slot.pnl,
        trades: slot.trades,
        winRate: slot.trades ? (slot.wins / slot.trades) * 100 : 0,
        avgR: slot.trades ? slot.rSum / slot.trades : 0,
      }));
  }, [chronological]);

  const topWinners = useMemo(() => [...sortedRows].sort((a, b) => b.pnl - a.pnl).slice(0, 8), [sortedRows]);
  const topLosers = useMemo(() => [...sortedRows].sort((a, b) => a.pnl - b.pnl).slice(0, 8), [sortedRows]);
  const expectancy = useMemo(() => {
    const pnls = sortedRows.map((trade) => trade.pnl);
    const rs = sortedRows.map((trade) => trade.r);
    const winsOnly = pnls.filter((value) => value > 0);
    const lossesOnly = pnls.filter((value) => value < 0);
    const avgWin = winsOnly.length ? winsOnly.reduce((a, b) => a + b, 0) / winsOnly.length : 0;
    const avgLoss = lossesOnly.length ? lossesOnly.reduce((a, b) => a + b, 0) / lossesOnly.length : 0;
    return {
      expectancyR: sortedRows.length ? rs.reduce((a, b) => a + b, 0) / sortedRows.length : 0,
      avgWin,
      avgLoss,
      payoffRatio: avgLoss < 0 ? avgWin / Math.abs(avgLoss) : 0,
      maxDrawdown: Math.abs(drawdownValues.reduce((acc, value) => Math.min(acc, value), 0)),
    };
  }, [drawdownValues, sortedRows]);

  return (
    <div className="space-y-5">
      <div className="glass-panel p-5">
        <div className="section-kicker">Performance Intelligence</div>
        <h2 className="mt-2 text-xl font-semibold text-white">PnL, attribution, and execution quality at multi-timeframe depth</h2>
        <p className="mt-3 text-sm leading-6 text-slate-300/75">{snapshot.performance.summary}</p>
      </div>

      {verdict ? (
        <div className="glass-panel p-5">
          <div className="section-kicker">Backtest Verdict</div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className={`rounded-full border px-3 py-1 text-[0.62rem] uppercase tracking-[0.22em] ${toneClasses[toneFromStatus(verdict.status)]}`}>
              {verdict.headline}
            </span>
            <span className={`rounded-full border px-3 py-1 text-[0.62rem] uppercase tracking-[0.22em] ${toneClasses[toneFromStatus(verdict.status)]}`}>
              {verdict.recommendation.replace(/_/g, " ")}
            </span>
          </div>
          <p className="mt-3 text-sm leading-6 text-slate-300/75">{verdict.summary}</p>

          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-4">
              <div className="text-[0.62rem] uppercase tracking-[0.18em] text-slate-400">Return</div>
              <div className="mt-2 text-lg font-semibold text-white">{fmtRatioPct(verdict.stats?.returnPct)}</div>
              <div className="mt-1 text-xs text-slate-500">{fmtSignedUsd(verdict.stats?.totalPnl ?? 0)}</div>
            </div>
            <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-4">
              <div className="text-[0.62rem] uppercase tracking-[0.18em] text-slate-400">Profit Factor</div>
              <div className="mt-2 text-lg font-semibold text-white">{fmtMaybeNumber(verdict.stats?.profitFactor, 2)}</div>
              <div className="mt-1 text-xs text-slate-500">gross win / gross loss</div>
            </div>
            <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-4">
              <div className="text-[0.62rem] uppercase tracking-[0.18em] text-slate-400">Max Drawdown</div>
              <div className="mt-2 text-lg font-semibold text-white">{fmtRatioPct(verdict.stats?.maxDrawdownPct)}</div>
              <div className="mt-1 text-xs text-slate-500">from starting equity</div>
            </div>
            <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-4">
              <div className="text-[0.62rem] uppercase tracking-[0.18em] text-slate-400">Positive Periods</div>
              <div className="mt-2 text-lg font-semibold text-white">{fmtRatioPct(verdict.stats?.positivePeriodShare)}</div>
              <div className="mt-1 text-xs text-slate-500">
                {verdict.stats?.periodCount ?? 0} {verdict.stats?.positivePeriodBasis ?? "periods"}
              </div>
            </div>
          </div>

          <div className="mt-4 grid gap-3 lg:grid-cols-2">
            {verdict.checks.map((check) => (
              <div key={check.label} className="rounded-3xl border border-white/10 bg-white/[0.03] p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="text-sm font-semibold text-white">{check.label}</div>
                  <span className={`rounded-full border px-2 py-1 text-[0.58rem] uppercase tracking-[0.18em] ${toneClasses[toneFromStatus(check.status)]}`}>
                    {check.status}
                  </span>
                </div>
                <div className="mt-3 text-sm text-slate-200">
                  <span className="font-mono text-white">{check.value}</span>
                  <span className="ml-2 text-slate-500">target {check.threshold}</span>
                </div>
                <p className="mt-2 text-xs leading-5 text-slate-400">{check.detail}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {snapshot.performance.kpis.map((metric) => (
          <MetricCard key={metric.label} metric={metric} />
        ))}
      </div>

      <div className="glass-panel p-5">
        <div className="section-kicker">Trade Lens</div>
        <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-7">
          <label className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2">
            <div className="text-[0.62rem] uppercase tracking-[0.18em] text-slate-400">Lookback</div>
            <select value={lookback} onChange={(event) => setLookback(event.target.value as Lookback)} className="mt-1 w-full bg-transparent text-sm text-white outline-none">
              <option value="all">All</option>
              <option value="30d">30d</option>
              <option value="90d">90d</option>
              <option value="180d">180d</option>
              <option value="365d">365d</option>
            </select>
          </label>
          <label className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2">
            <div className="text-[0.62rem] uppercase tracking-[0.18em] text-slate-400">Side</div>
            <select value={sideFilter} onChange={(event) => setSideFilter(event.target.value as "all" | "long" | "short")} className="mt-1 w-full bg-transparent text-sm text-white outline-none">
              <option value="all">All</option>
              <option value="long">Long</option>
              <option value="short">Short</option>
            </select>
          </label>
          <label className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2">
            <div className="text-[0.62rem] uppercase tracking-[0.18em] text-slate-400">Tier</div>
            <select value={tierFilter} onChange={(event) => setTierFilter(event.target.value)} className="mt-1 w-full bg-transparent text-sm text-white outline-none">
              {tierOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </label>
          <label className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2">
            <div className="text-[0.62rem] uppercase tracking-[0.18em] text-slate-400">Model</div>
            <select value={modelFilter} onChange={(event) => setModelFilter(event.target.value)} className="mt-1 w-full bg-transparent text-sm text-white outline-none">
              {modelOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </label>
          <label className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2">
            <div className="text-[0.62rem] uppercase tracking-[0.18em] text-slate-400">Session</div>
            <select value={sessionFilter} onChange={(event) => setSessionFilter(event.target.value)} className="mt-1 w-full bg-transparent text-sm text-white outline-none">
              {sessionOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </label>
          <label className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2">
            <div className="text-[0.62rem] uppercase tracking-[0.18em] text-slate-400">Sort</div>
            <select value={sortBy} onChange={(event) => setSortBy(event.target.value as SortKey)} className="mt-1 w-full bg-transparent text-sm text-white outline-none">
              <option value="entryTs">Entry Time</option>
              <option value="exitTs">Exit Time</option>
              <option value="pnl">PnL</option>
              <option value="r">R Multiple</option>
              <option value="holdMinutes">Hold</option>
              <option value="fees">Fees</option>
              <option value="slippageBps">Slippage</option>
            </select>
          </label>
          <label className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2">
            <div className="text-[0.62rem] uppercase tracking-[0.18em] text-slate-400">Search</div>
            <input value={searchText} onChange={(event) => setSearchText(event.target.value)} placeholder="trade / reason / regime" className="mt-1 w-full bg-transparent text-sm text-white placeholder:text-slate-500 outline-none" />
          </label>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button
            onClick={() => {
              setLookback("all");
              setSideFilter("all");
              setTierFilter("all");
              setModelFilter("all");
              setSessionFilter("all");
              setSearchText("");
              setSortBy("entryTs");
              setSortDir("desc");
              setPage(0);
            }}
            className="rounded-full border border-cyan/30 bg-cyan/10 px-3 py-1 text-xs uppercase tracking-[0.2em] text-cyan transition hover:border-cyan/50"
          >
            Reset Lens
          </button>
          <button
            onClick={() => setSortDir((current) => (current === "asc" ? "desc" : "asc"))}
            className="rounded-full border border-white/15 bg-white/[0.03] px-3 py-1 text-xs uppercase tracking-[0.2em] text-slate-300 transition hover:border-cyan/35 hover:text-cyan"
          >
            Order: {sortDir === "asc" ? "Ascending" : "Descending"}
          </button>
          <div className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs uppercase tracking-[0.16em] text-slate-400">
            Showing {sortedRows.length} / {allRows.length} trades
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <ChartCard title="Equity Curve (Filtered)" subtitle="Hover for exact timestamp and equity level">
          <InteractiveLineSurface
            values={equityValues}
            labels={sequenceLabels}
            color="#52d7ff"
            fill="rgba(82, 215, 255, 0.14)"
            valueFormatter={(value) => fmtUsd(value)}
          />
        </ChartCard>
        <ChartCard title="Drawdown Surface (Filtered)" subtitle="Underwater profile from running equity highs">
          <InteractiveLineSurface
            values={drawdownValues}
            labels={sequenceLabels}
            color="#ff6b88"
            fill="rgba(255, 107, 136, 0.16)"
            valueFormatter={(value) => fmtSignedUsd(value)}
          />
        </ChartCard>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <ChartCard title="PnL Sequence" subtitle="Chronological trade-by-trade result">
          <InteractiveBarSurface
            values={pnlSequence}
            labels={sequenceLabels}
            positiveColor="#2ae6b8"
            negativeColor="#ff6b88"
            valueFormatter={(value) => fmtSignedUsd(value)}
          />
        </ChartCard>
        <ChartCard title="R Sequence" subtitle="Risk-adjusted distribution over time">
          <InteractiveBarSurface
            values={rSequence}
            labels={sequenceLabels}
            positiveColor="#52d7ff"
            negativeColor="#f6b63c"
            valueFormatter={(value) => `${value.toFixed(2)}R`}
          />
        </ChartCard>
        <ChartCard title="Hourly Edge (UTC)" subtitle="PnL attribution by execution hour">
          <BarSurface
            values={byHour.map((row) => row.pnl)}
            labels={byHour.map((row) => row.label.slice(0, 2))}
            positiveColor="#2ae6b8"
            negativeColor="#ff6b88"
          />
        </ChartCard>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {snapshot.performance.periods.map((period) => (
          <div key={period.label} className="metric-card p-5">
            <div className="section-kicker">{period.label}</div>
            <div className={`mt-4 text-2xl font-semibold ${period.pnl >= 0 ? "text-teal" : "text-rose"}`}>
              {fmtSignedUsd(period.pnl)}
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2 text-sm">
              <div className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-slate-200">{period.trades} trades</div>
              <div className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-slate-200">{period.winRate.toFixed(1)}% WR</div>
              <div className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-slate-200">{period.avgR.toFixed(2)}R</div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <BucketPanel title="By Asset" rows={byAsset} />
        <BucketPanel title="By Tier" rows={byTier} />
        <BucketPanel title="By Model" rows={byModel} />
        <BucketPanel title="By Session" rows={bySession} />
        <BucketPanel title="By Regime" rows={byRegime} />
        <BucketPanel title="By Hold Bucket" rows={byHold} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <ChartCard title="Monthly PnL Timeline" subtitle="Aggregated month-on-month edge profile">
          <BarSurface
            values={monthlyTimeline.map((row) => row.pnl)}
            labels={monthlyTimeline.map((row) => row.label.slice(5))}
            positiveColor="#2ae6b8"
            negativeColor="#ff6b88"
          />
        </ChartCard>
        <ChartCard title="Weekday Edge" subtitle="Execution quality by day-of-week">
          <BarSurface
            values={weekdayRows.map((row) => row.pnl)}
            labels={weekdayRows.map((row) => row.label)}
            positiveColor="#52d7ff"
            negativeColor="#ff6b88"
          />
        </ChartCard>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard metric={{ label: "Expectancy", value: `${expectancy.expectancyR.toFixed(3)}R`, tone: expectancy.expectancyR >= 0 ? "teal" : "rose", delta: "avg R / trade" }} />
        <MetricCard metric={{ label: "Avg Win", value: fmtSignedUsd(expectancy.avgWin), tone: "teal", delta: "winning trades" }} />
        <MetricCard metric={{ label: "Avg Loss", value: fmtSignedUsd(expectancy.avgLoss), tone: "rose", delta: "losing trades" }} />
        <MetricCard metric={{ label: "Payoff", value: expectancy.payoffRatio.toFixed(2), tone: expectancy.payoffRatio >= 1 ? "teal" : "amber", delta: "avg win / avg loss" }} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="glass-panel p-5">
          <div className="section-kicker">Top Winners</div>
          <div className="mt-3 space-y-2">
            {topWinners.length ? topWinners.map((trade) => (
              <button
                key={`winner-${trade.tradeId}-${trade.entryTs}`}
                onClick={() => setSelectedTradeId(trade.tradeId)}
                className={`w-full rounded-2xl border px-3 py-2 text-left transition hover:border-teal/40 hover:bg-teal/10 ${
                  trade.tradeId === selectedTradeId ? "border-cyan/40 bg-cyan/10" : "border-white/10 bg-white/[0.03]"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="font-mono text-xs text-slate-300">{trade.tradeId}</div>
                  <div className="text-sm text-teal">{fmtSignedUsd(trade.pnl)}</div>
                </div>
                <div className="mt-1 text-xs text-slate-400">{trade.asset} • {trade.model || "unknown"} • {trade.r.toFixed(2)}R</div>
              </button>
            )) : <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3 text-sm text-slate-400">No winners in current filter.</div>}
          </div>
        </div>
        <div className="glass-panel p-5">
          <div className="section-kicker">Top Losers</div>
          <div className="mt-3 space-y-2">
            {topLosers.length ? topLosers.map((trade) => (
              <button
                key={`loser-${trade.tradeId}-${trade.entryTs}`}
                onClick={() => setSelectedTradeId(trade.tradeId)}
                className={`w-full rounded-2xl border px-3 py-2 text-left transition hover:border-rose/40 hover:bg-rose/10 ${
                  trade.tradeId === selectedTradeId ? "border-cyan/40 bg-cyan/10" : "border-white/10 bg-white/[0.03]"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="font-mono text-xs text-slate-300">{trade.tradeId}</div>
                  <div className="text-sm text-rose">{fmtSignedUsd(trade.pnl)}</div>
                </div>
                <div className="mt-1 text-xs text-slate-400">{trade.asset} • {trade.model || "unknown"} • {trade.r.toFixed(2)}R</div>
              </button>
            )) : <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3 text-sm text-slate-400">No losers in current filter.</div>}
          </div>
        </div>
      </div>

      <div className="glass-panel overflow-hidden">
        <div className="border-b border-white/10 px-5 py-4">
          <div className="section-kicker">Trade Table</div>
          <div className="mt-2 text-sm text-slate-300">Click any row for full drilldown. Table reflects current filters and sort settings.</div>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="sticky top-0 z-10 border-b border-white/10 bg-[#08131f]/95 text-xs uppercase tracking-[0.18em] text-slate-400 backdrop-blur">
              <tr>
                {["Trade", "Asset", "Side", "Tier", "Model", "Session", "Regime", "PnL", "R", "Hold", "Fees", "Slip", "Entry", "Exit", "Reason"].map((header) => (
                  <th key={header} className="px-3 py-3 font-medium">{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pageRows.map((trade) => (
                <tr
                  key={`${trade.tradeId}-${trade.entryTs}`}
                  onClick={() => setSelectedTradeId(trade.tradeId)}
                  className={`cursor-pointer border-b border-white/5 transition hover:bg-cyan/5 ${
                    trade.tradeId === selectedTradeId ? "bg-cyan/10" : ""
                  }`}
                >
                  <td className="px-3 py-3 font-mono text-xs text-slate-300">{trade.tradeId}</td>
                  <td className="px-3 py-3 font-medium text-white">{trade.asset}</td>
                  <td className={`px-3 py-3 ${trade.side === "short" ? "text-rose" : "text-teal"}`}>{trade.side ?? "long"}</td>
                  <td className="px-3 py-3 text-amber">{trade.tier}</td>
                  <td className="px-3 py-3 text-slate-200">{trade.model || "-"}</td>
                  <td className="px-3 py-3 text-slate-300">{trade.session || "-"}</td>
                  <td className="px-3 py-3 text-slate-300">{trade.regime || "-"}</td>
                  <td className={`px-3 py-3 font-medium ${trade.pnl >= 0 ? "text-teal" : "text-rose"}`}>{fmtSignedUsd(trade.pnl)}</td>
                  <td className="px-3 py-3 text-cyan">{trade.r.toFixed(2)}R</td>
                  <td className="px-3 py-3 text-slate-300">{trade.holdMinutes ? `${Math.round(trade.holdMinutes)}m` : "-"}</td>
                  <td className="px-3 py-3 text-amber">{trade.fees ? fmtUsd(trade.fees) : "-"}</td>
                  <td className="px-3 py-3 text-slate-300">{trade.slippageBps ? `${trade.slippageBps.toFixed(2)} bps` : "-"}</td>
                  <td className="px-3 py-3 text-slate-400">{fmtTs(trade.entryTs)}</td>
                  <td className="px-3 py-3 text-slate-400">{fmtTs(trade.exitTs || "")}</td>
                  <td className="max-w-[360px] truncate px-3 py-3 text-slate-400" title={trade.reason}>{trade.reason}</td>
                </tr>
              ))}
              {pageRows.length === 0 ? (
                <tr>
                  <td colSpan={15} className="px-4 py-6 text-center text-slate-400">No trade rows available for the current lens.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between border-t border-white/10 px-5 py-3 text-xs text-slate-400">
          <div>Page {Math.min(page + 1, pageCount)} / {pageCount}</div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((current) => Math.max(0, current - 1))}
              disabled={page <= 0}
              className="rounded-full border border-white/15 px-3 py-1 transition disabled:cursor-not-allowed disabled:opacity-40 hover:border-cyan/40 hover:text-cyan"
            >
              Prev
            </button>
            <button
              onClick={() => setPage((current) => Math.min(pageCount - 1, current + 1))}
              disabled={page >= pageCount - 1}
              className="rounded-full border border-white/15 px-3 py-1 transition disabled:cursor-not-allowed disabled:opacity-40 hover:border-cyan/40 hover:text-cyan"
            >
              Next
            </button>
          </div>
        </div>
      </div>

      {selectedTrade ? (
        <div className="glass-panel p-5">
          <div className="section-kicker">Selected Trade Drilldown</div>
          <div className="mt-3 grid gap-3 md:grid-cols-5">
            <DrillValue label="Trade" value={selectedTrade.tradeId} />
            <DrillValue label="Asset" value={selectedTrade.asset} />
            <DrillValue label="Side" value={selectedTrade.side || "long"} tone={selectedTrade.side === "short" ? "rose" : "teal"} />
            <DrillValue label="Tier" value={selectedTrade.tier} />
            <DrillValue label="Model" value={selectedTrade.model || "-"} />
            <DrillValue label="Session" value={selectedTrade.session || "-"} />
            <DrillValue label="Regime" value={selectedTrade.regime || "-"} />
            <DrillValue label="PnL" value={fmtSignedUsd(selectedTrade.pnl)} tone={selectedTrade.pnl >= 0 ? "teal" : "rose"} />
            <DrillValue label="R Multiple" value={`${selectedTrade.r.toFixed(2)}R`} tone="cyan" />
            <DrillValue label="Notional" value={selectedTrade.notional ? fmtUsd(selectedTrade.notional) : "-"} />
            <DrillValue label="Risk USD" value={selectedTrade.riskUsd ? fmtUsd(selectedTrade.riskUsd) : "-"} />
            <DrillValue label="Hold" value={selectedTrade.holdMinutes ? `${Math.round(selectedTrade.holdMinutes)}m` : "-"} />
            <DrillValue label="Fees" value={selectedTrade.fees ? fmtUsd(selectedTrade.fees) : "-"} />
            <DrillValue label="Slippage" value={selectedTrade.slippageBps ? `${selectedTrade.slippageBps.toFixed(2)} bps` : "-"} />
            <DrillValue label="MAE / MFE" value={`${(selectedTrade.mae || 0).toFixed(2)} / ${(selectedTrade.mfe || 0).toFixed(2)}`} />
            <DrillValue label="Entry" value={fmtTs(selectedTrade.entryTs)} />
            <DrillValue label="Exit" value={fmtTs(selectedTrade.exitTs || "")} />
            <DrillValue label="Reason" value={selectedTrade.reason} />
          </div>
        </div>
      ) : null}
    </div>
  );
}

function InsightsPanel({ summary, trace, reasoning }: { summary: string; trace: InsightNode[]; reasoning?: ReasoningTree }) {
  const [selectedTrace, setSelectedTrace] = useState<string | null>(trace[0]?.label ?? null);
  const traceSeries = useMemo(
    () =>
      trace.map((node, index) => {
        const numeric = Number.parseFloat(String(node.value).replace(/[^\d.-]/g, ""));
        if (Number.isFinite(numeric)) return numeric;
        return index + 1;
      }),
    [trace],
  );
  const activeTrace = trace.find((node) => node.label === selectedTrace) ?? trace[0];

  useEffect(() => {
    if (!trace.length) return;
    if (!selectedTrace || !trace.some((node) => node.label === selectedTrace)) {
      setSelectedTrace(trace[0].label);
    }
  }, [selectedTrace, trace]);

  return (
    <div className="space-y-5">
      <div className="glass-panel p-5">
        <div className="section-kicker">Insights</div>
        <h2 className="mt-2 text-xl font-semibold text-white">State inspection, not indicator clutter</h2>
        <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-300/75">{summary}</p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {trace.map((node) => (
          <motion.button
            key={node.label}
            whileHover={{ y: -4 }}
            onClick={() => setSelectedTrace(node.label)}
            className={`metric-card w-full p-5 text-left ${selectedTrace === node.label ? "border-cyan/45 bg-cyan/10" : ""}`}
          >
            <div className={`inline-flex rounded-full border px-3 py-1 text-xs uppercase tracking-[0.2em] ${toneClasses[node.tone]}`}>
              {node.label}
            </div>
            <div className="mt-4 text-2xl font-semibold text-white">{node.value}</div>
            <p className="mt-3 text-sm leading-6 text-slate-300/80">{node.detail}</p>
          </motion.button>
        ))}
      </div>
      <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <ChartCard
          title="Feature State Pulse"
          subtitle="Relative movement across selected insight dimensions"
        >
          <LineSurface values={traceSeries} color="#2ae6b8" fill="rgba(42, 230, 184, 0.14)" />
        </ChartCard>
        <div className="glass-panel p-5">
          <div className="section-kicker">Trace Drilldown</div>
          {activeTrace ? (
            <>
              <h3 className="mt-2 text-lg font-semibold text-white">{activeTrace.label}</h3>
              <div className="mt-3 text-2xl font-semibold text-cyan">{activeTrace.value}</div>
              <p className="mt-3 text-sm leading-6 text-slate-300/80">{activeTrace.detail}</p>
            </>
          ) : (
            <div className="mt-3 text-sm text-slate-400">No trace details available.</div>
          )}
        </div>
      </div>
      <ReasoningTreePanel
        title="Alert Reasoning Tree"
        subtitle="The actual structured decision payload behind the current alert selection."
        reasoning={reasoning}
      />
    </div>
  );
}

function RegimePanel({ snapshot }: { snapshot: TerminalSnapshot }) {
  const [selectedState, setSelectedState] = useState<string | null>(snapshot.regime.states[0]?.name ?? null);
  const activeState = snapshot.regime.states.find((state) => state.name === selectedState) ?? snapshot.regime.states[0];
  const probabilitySeries = snapshot.regime.states.map((state) => state.probability * 100);

  useEffect(() => {
    if (!snapshot.regime.states.length) return;
    if (!selectedState || !snapshot.regime.states.some((state) => state.name === selectedState)) {
      setSelectedState(snapshot.regime.states[0].name);
    }
  }, [selectedState, snapshot.regime.states]);

  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard metric={{ label: "Current Regime", value: snapshot.regime.current, tone: "cyan", delta: "12h anchor" }} />
        <MetricCard metric={{ label: "Persistence", value: `${snapshot.regime.persistence}%`, tone: "teal", delta: "stability" }} />
        <MetricCard metric={{ label: "Transition Risk", value: `${snapshot.regime.transitionRisk}%`, tone: "amber", delta: "state change risk" }} />
      </div>
      <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <ChartCard
          title="Regime Probability Surface"
          subtitle="Relative state probabilities used for context conditioning"
        >
          <BarSurface values={probabilitySeries} labels={snapshot.regime.states.map((state) => state.name.split(" ")[0])} positiveColor="#52d7ff" negativeColor="#52d7ff" />
        </ChartCard>
        <div className="glass-panel p-5">
          <div className="section-kicker">Selected Regime</div>
          {activeState ? (
            <>
              <h3 className="mt-2 text-lg font-semibold text-white">{activeState.name}</h3>
              <div className="mt-2 text-2xl font-semibold text-cyan">{Math.round(activeState.probability * 100)}%</div>
              <p className="mt-3 text-sm leading-6 text-slate-300/80">{activeState.description}</p>
            </>
          ) : (
            <div className="mt-3 text-sm text-slate-400">No regime selected.</div>
          )}
        </div>
      </div>
      <div className="glass-panel p-5">
        <div className="section-kicker">Regime Briefings</div>
        <h2 className="mt-2 text-xl font-semibold text-white">State probabilities with institutional readability</h2>
        <div className="mt-5 space-y-4">
          {snapshot.regime.states.map((state) => (
            <button
              key={state.name}
              onClick={() => setSelectedState(state.name)}
              className={`w-full rounded-2xl border bg-white/[0.03] p-4 text-left transition hover:border-cyan/35 hover:bg-cyan/5 ${selectedState === state.name ? "border-cyan/40" : "border-white/8"}`}
            >
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="font-medium text-white">{state.name}</div>
                  <div className="mt-1 text-sm text-slate-400">{state.description}</div>
                </div>
                <div className="min-w-[140px] text-right">
                  <div className="font-mono text-lg text-cyan">{Math.round(state.probability * 100)}%</div>
                  <div className="mt-2 h-2 rounded-full bg-white/8">
                    <div className="h-full rounded-full bg-gradient-to-r from-cyan to-teal" style={{ width: `${Math.max(6, state.probability * 100)}%` }} />
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function SignalsPanel({
  summary,
  candidates,
  selectedSignalId,
  onSelectSignal,
  reasoning,
}: {
  summary: string;
  candidates: SignalCandidate[];
  selectedSignalId: string | null;
  onSelectSignal: (signalId: string) => void;
  reasoning?: ReasoningTree;
}) {
  const scatterPoints = candidates.map((candidate) => ({
    x: candidate.confluence * 100,
    y: candidate.hazard * 100,
    label: candidate.asset,
    tone: candidate.side === "long" ? ("teal" as const) : ("rose" as const),
  }));

  return (
    <div className="space-y-5">
      <div className="glass-panel p-5">
        <div className="section-kicker">Signal Intelligence</div>
        <h2 className="mt-2 text-xl font-semibold text-white">Comparative ranking across the active opportunity surface</h2>
        <p className="mt-3 text-sm leading-6 text-slate-300/75">{summary}</p>
      </div>
      <ChartCard
        title="Confluence vs Hazard Map"
        subtitle="Upper-left is highest quality (high confluence, low hazard)"
      >
        <ScatterSurface points={scatterPoints} />
      </ChartCard>
      <div className="glass-panel overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-white/10 bg-white/[0.03] text-xs uppercase tracking-[0.18em] text-slate-400">
              <tr>
                {["Asset", "Side", "Tier", "Confluence", "EVR", "Flow 1h", "Hazard", "Reason"].map((header) => (
                  <th key={header} className="px-4 py-4 font-medium">{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {candidates.map((candidate) => (
                <tr
                  key={candidate.id}
                  onClick={() => onSelectSignal(candidate.id)}
                  className={`cursor-pointer border-b border-white/5 transition hover:bg-cyan/5 ${
                    selectedSignalId === candidate.id ? "bg-cyan/10" : ""
                  }`}
                >
                  <td className="px-4 py-4 font-medium text-white">{candidate.asset}</td>
                  <td className={`px-4 py-4 ${candidate.side === "long" ? "text-teal" : "text-rose"}`}>
                    {candidate.side === "long" ? <ArrowUpRight className="inline h-4 w-4" /> : <ArrowDownRight className="inline h-4 w-4" />} {candidate.side}
                  </td>
                  <td className="px-4 py-4 text-amber">{candidate.tier}</td>
                  <td className="px-4 py-4 text-cyan">{candidate.confluence.toFixed(2)}</td>
                  <td className="px-4 py-4 text-slate-200">{candidate.evr.toFixed(2)}</td>
                  <td className="px-4 py-4 text-teal">{candidate.flow1h.toFixed(2)}</td>
                  <td className="px-4 py-4 text-rose">{candidate.hazard.toFixed(2)}</td>
                  <td className="px-4 py-4 text-slate-400">{candidate.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <ReasoningTreePanel
        title="Selected Signal Vector"
        subtitle="Click any ranked candidate to inspect the exact nested decision vector that generated the alert."
        reasoning={reasoning}
      />
    </div>
  );
}

function ConfluencePanel({
  snapshot,
  candidates,
  selectedSignalId,
  onSelectSignal,
  reasoning,
}: {
  snapshot: TerminalSnapshot;
  candidates: SignalCandidate[];
  selectedSignalId: string | null;
  onSelectSignal: (signalId: string) => void;
  reasoning?: ReasoningTree;
}) {
  const selectedCandidate = candidates.find((candidate) => candidate.id === selectedSignalId) ?? candidates[0];
  const envelope = getObject(reasoning);
  const decision = getObject(envelope?.decision);
  const reasoningNode = getObject(envelope?.reasoning);
  const ml = getObject(reasoningNode?.ml);
  const hazardNode = getObject(reasoningNode?.hazard);
  const finalDecision = getObject(reasoningNode?.final_decision);
  const context = getObject(reasoningNode?.context);
  const specialists = [
    { label: "Liquidity", value: pickNumber(ml?.p_liq_flow, ml?.prob_liq_flow) ?? 0.0 },
    { label: "BOS", value: pickNumber(ml?.p_bos_cont, ml?.prob_bos_cont) ?? 0.0 },
    { label: "Flow 1h", value: pickNumber(ml?.p_flow_1h, ml?.prob_flow_1h, selectedCandidate?.flow1h) ?? 0.0 },
  ];
  const ruleConfluence = pickNumber(
    finalDecision?.confluence,
    decision?.confluence,
    selectedCandidate?.confluence,
    0.0,
  ) ?? 0.0;
  const mlConfluence = pickNumber(
    ml?.prob_confluence,
    ml?.p_confluence,
    finalDecision?.prob_confluence,
    selectedCandidate?.confluence,
    ruleConfluence,
  ) ?? ruleConfluence;
  const hazard = pickNumber(
    hazardNode?.hazard_score,
    ml?.hazard_score,
    selectedCandidate?.hazard,
    0.0,
  ) ?? 0.0;
  const evr = pickNumber(finalDecision?.evr, decision?.evr, selectedCandidate?.evr, 0.0) ?? 0.0;
  const flow1h = pickNumber(ml?.p_flow_1h, ml?.prob_flow_1h, selectedCandidate?.flow1h, 0.0) ?? 0.0;
  const routeEffective = pickString(
    envelope?.routing_mode_effective,
    envelope?.effective_routing_mode,
    reasoningNode?.routing_mode_effective,
    context?.routing_mode_effective,
    envelope?.inference_source_mode,
  ) ?? "unreported";
  const routeRequested = pickString(
    envelope?.routing_mode_requested,
    reasoningNode?.routing_mode_requested,
    context?.routing_mode_requested,
  ) ?? "tree";
  const challengerMode = pickString(
    envelope?.challenger_mode,
    reasoningNode?.challenger_mode,
    context?.challenger_mode,
  ) ?? "tcn";
  const side = pickString(decision?.side, selectedCandidate?.side) ?? "long";
  const tier = pickString(finalDecision?.tier, decision?.tier, selectedCandidate?.tier) ?? "-";
  const regime = pickString(finalDecision?.regime, decision?.regime, context?.regime, selectedCandidate?.regime) ?? "unknown";
  const operatorConfluence = pickNumber(selectedCandidate?.confluence, finalDecision?.confluence, decision?.confluence, ruleConfluence, mlConfluence, 0.0) ?? 0.0;
  const operatorGuide = buildAlertOperatorGuide({
    tier,
    confluence: operatorConfluence,
    evr,
    hazard,
    flow1h,
  });
  const verdictOptions = [
    {
      label: "Green Light",
      emoji: "🟢",
      icon: CheckCircle2,
      tone: "teal" as const,
      blurb: "System-approved process trade.",
    },
    {
      label: "Caution",
      emoji: "🟠",
      icon: AlertTriangle,
      tone: "amber" as const,
      blurb: "Valid, but more conditional.",
    },
    {
      label: "Pass",
      emoji: "🔴",
      icon: CircleX,
      tone: "rose" as const,
      blurb: "Stand down unless context changes.",
    },
  ];
  const confluenceGap = mlConfluence - ruleConfluence;
  const stackVector = [
    ruleConfluence * 100,
    mlConfluence * 100,
    flow1h * 100,
    (1 - Math.max(0, Math.min(1, hazard))) * 100,
    ...specialists.map((item) => item.value * 100),
  ];
  const stackLabels = ["Rule", "ML", "Flow 1h", "Hazard Inv.", "Liquidity", "BOS"];
  const scatterPoints = candidates.map((candidate) => ({
    x: candidate.confluence * 100,
    y: candidate.hazard * 100,
    label: candidate.asset,
    tone: candidate.side === "long" ? ("teal" as const) : ("rose" as const),
  }));

  return (
    <div className="space-y-5">
      <div className="glass-panel p-5">
        <div className="section-kicker">Confluence Studio</div>
        <h2 className="mt-2 text-xl font-semibold text-white">Stack agreement and execution quality in one desk</h2>
        <p className="mt-3 text-sm leading-6 text-slate-300/75">
          Confluence is kept separate here so you can read the decision stack cleanly without disturbing the original terminal shell.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <MetricCard metric={{ label: "Rule Confluence", value: ruleConfluence.toFixed(2), tone: ruleConfluence >= 0.75 ? "teal" : ruleConfluence >= 0.6 ? "amber" : "rose", delta: `${tier} tier` }} />
        <MetricCard metric={{ label: "ML Confluence", value: mlConfluence.toFixed(2), tone: mlConfluence >= 0.75 ? "teal" : mlConfluence >= 0.6 ? "amber" : "rose", delta: `${confluenceGap >= 0 ? "+" : ""}${confluenceGap.toFixed(2)} vs rule` }} />
        <MetricCard metric={{ label: "EVR", value: evr.toFixed(2), tone: evr >= 2 ? "teal" : evr >= 1.5 ? "amber" : "rose", delta: regime }} />
        <MetricCard metric={{ label: "Flow 1h", value: flow1h.toFixed(2), tone: flow1h >= 0.7 ? "teal" : flow1h >= 0.58 ? "amber" : "rose", delta: side }} />
        <MetricCard metric={{ label: "Hazard", value: hazard.toFixed(2), tone: hazard <= 0.2 ? "teal" : hazard <= 0.32 ? "amber" : "rose", delta: "lower is better" }} />
        <MetricCard metric={{ label: "Route", value: routeEffective, tone: routeEffective === "tcn" ? "teal" : routeEffective === "hybrid_explicit" ? "cyan" : "amber", delta: `req ${routeRequested} • chall ${challengerMode}` }} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <ChartCard
          title="Confluence Stack Vector"
          subtitle="Rule, ML, flow, and specialist support aligned on the same scale"
        >
          <InteractiveBarSurface
            values={stackVector}
            labels={stackLabels}
            positiveColor="#52d7ff"
            negativeColor="#ff6b88"
            valueFormatter={(value) => `${value.toFixed(1)}%`}
          />
        </ChartCard>
        <ChartCard
          title="Candidate Quality Map"
          subtitle="Upper-left remains best: high confluence, low hazard"
        >
          <ScatterSurface points={scatterPoints} />
        </ChartCard>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.15fr_0.95fr]">
        <div className="glass-panel overflow-hidden">
          <div className="border-b border-white/10 px-5 py-4">
            <div className="section-kicker">Signal Ladder</div>
            <div className="mt-2 text-sm text-slate-300">Select the candidate whose confluence stack you want to inspect.</div>
          </div>
          <div className="divide-y divide-white/6">
            {candidates.map((candidate) => {
              const active = candidate.id === selectedCandidate?.id;
              return (
                <button
                  key={candidate.id}
                  onClick={() => onSelectSignal(candidate.id)}
                  className={`w-full px-5 py-4 text-left transition hover:bg-cyan/5 ${active ? "bg-cyan/10" : ""}`}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="font-medium text-white">{candidate.asset}</div>
                      <div className="mt-1 text-xs uppercase tracking-[0.18em] text-slate-400">
                        {candidate.side} • {candidate.tier} • {candidate.regime}
                      </div>
                      <div className="mt-2 text-sm text-slate-300/75">{candidate.reason}</div>
                    </div>
                    <div className="min-w-[132px]">
                      <div className="flex items-center justify-between text-xs uppercase tracking-[0.16em] text-slate-400">
                        <span>Conf</span>
                        <span className="text-cyan">{candidate.confluence.toFixed(2)}</span>
                      </div>
                      <div className="mt-2 h-2 rounded-full bg-white/8">
                        <div className="h-full rounded-full bg-gradient-to-r from-cyan to-teal" style={{ width: `${candidate.confluence * 100}%` }} />
                      </div>
                      <div className="mt-3 flex items-center justify-between text-xs uppercase tracking-[0.16em] text-slate-400">
                        <span>Haz</span>
                        <span className="text-rose">{candidate.hazard.toFixed(2)}</span>
                      </div>
                      <div className="mt-2 h-2 rounded-full bg-white/8">
                        <div className="h-full rounded-full bg-gradient-to-r from-amber to-rose" style={{ width: `${candidate.hazard * 100}%` }} />
                      </div>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="space-y-4">
          <div className="glass-panel p-5">
            <div className="section-kicker">Alert Decision</div>
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <h3 className="text-lg font-semibold text-white">Operator verdict</h3>
              <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs uppercase tracking-[0.2em] ${toneClasses[operatorGuide.tone]}`}>
                <span>{operatorGuide.label === "Green Light" ? "🟢" : operatorGuide.label === "Caution" ? "🟠" : "🔴"}</span>
                <span>{operatorGuide.label}</span>
              </span>
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-300/80">{operatorGuide.summary}</p>
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              {verdictOptions.map((option) => {
                const active = option.label === operatorGuide.label;
                const Icon = option.icon;
                return (
                  <div
                    key={option.label}
                    className={`rounded-2xl border px-4 py-3 transition ${
                      active
                        ? `${toneClasses[option.tone]} shadow-[0_0_0_1px_rgba(255,255,255,0.04)]`
                        : "border-white/10 bg-white/[0.02] text-slate-300"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <Icon className="h-4 w-4" />
                      <div className="text-sm font-semibold">
                        {option.emoji} {option.label}
                      </div>
                    </div>
                    <div className="mt-2 text-xs leading-5 text-slate-300/75">{option.blurb}</div>
                  </div>
                );
              })}
            </div>
            <div className="mt-4 space-y-2">
              {operatorGuide.checks.map((check) => (
                <div key={check.label} className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-medium text-white">{check.label}</div>
                    <div className={`inline-flex rounded-full border px-2 py-1 text-[0.62rem] uppercase tracking-[0.18em] ${toneClasses[check.tone]}`}>
                      {check.status}
                    </div>
                  </div>
                  <div className="mt-2 text-sm leading-6 text-slate-300/75">{check.detail}</div>
                </div>
              ))}
            </div>
            <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3">
              <div className="text-xs uppercase tracking-[0.18em] text-slate-400">Default action</div>
              <div className="mt-2 text-sm leading-6 text-slate-200">{operatorGuide.action}</div>
            </div>
          </div>
          <div className="glass-panel p-5">
            <div className="section-kicker">Execution Read</div>
            <h3 className="mt-2 text-lg font-semibold text-white">{selectedCandidate?.asset ?? "No candidate selected"}</h3>
            <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-1">
              <NarrativePanel title="Requested Route" text={`Requested ${routeRequested}, effective ${routeEffective}, challenger ${challengerMode}.`} />
              <NarrativePanel title="Decision Surface" text={`Rule confluence ${ruleConfluence.toFixed(2)}, ML confluence ${mlConfluence.toFixed(2)}, EVR ${evr.toFixed(2)}, hazard ${hazard.toFixed(2)}.`} />
              <NarrativePanel title="Specialist Support" text={`Liquidity ${specialists[0].value.toFixed(2)}, BOS ${specialists[1].value.toFixed(2)}, Flow 1h ${flow1h.toFixed(2)}.`} />
            </div>
          </div>
          <ChartCard
            title="Selected Specialist Balance"
            subtitle="Current candidate only"
          >
            <InteractiveBarSurface
              values={[...specialists.map((item) => item.value * 100), mlConfluence * 100, hazard * 100]}
              labels={["Liquidity", "BOS", "Flow 1h", "ML Conf", "Hazard"]}
              positiveColor="#2ae6b8"
              negativeColor="#ff6b88"
              valueFormatter={(value) => `${value.toFixed(1)}%`}
            />
          </ChartCard>
        </div>
      </div>

      <ReasoningTreePanel
        title="Selected Confluence Payload"
        subtitle="Exact structured payload attached to the selected candidate."
        reasoning={reasoning}
      />

      <div className="glass-panel p-5">
        <div className="section-kicker">System Link</div>
        <div className="grid gap-4 md:grid-cols-3">
          <TelemetryRow icon={<Gauge className="h-4 w-4" />} label="Model Version" value={snapshot.meta.modelVersion} />
          <TelemetryRow icon={<ShieldCheck className="h-4 w-4" />} label="Transport" value={snapshot.meta.transport} />
          <TelemetryRow icon={<BrainCircuit className="h-4 w-4" />} label="Active Regime" value={snapshot.regime.current} />
        </div>
      </div>
    </div>
  );
}

function RiskPanel({ snapshot }: { snapshot: TerminalSnapshot }) {
  const [selectedGuardrail, setSelectedGuardrail] = useState<string | null>(snapshot.risk.guardrails[0]?.label ?? null);
  const riskPoints = [
    { axis: "Stress", value: snapshot.risk.stress },
    { axis: "Slippage", value: snapshot.risk.slippage },
    { axis: "Exposure", value: snapshot.risk.exposure },
    { axis: "Transition", value: snapshot.regime.transitionRisk },
  ];
  const activeGuardrail = snapshot.risk.guardrails.find((guardrail) => guardrail.label === selectedGuardrail) ?? snapshot.risk.guardrails[0];

  useEffect(() => {
    if (!snapshot.risk.guardrails.length) return;
    if (!selectedGuardrail || !snapshot.risk.guardrails.some((guardrail) => guardrail.label === selectedGuardrail)) {
      setSelectedGuardrail(snapshot.risk.guardrails[0].label);
    }
  }, [selectedGuardrail, snapshot.risk.guardrails]);

  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-3">
        <GaugeCard label="Stress" value={snapshot.risk.stress} />
        <GaugeCard label="Slippage" value={snapshot.risk.slippage} />
        <GaugeCard label="Exposure" value={snapshot.risk.exposure} />
      </div>
      <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <ChartCard
          title="Risk Geometry"
          subtitle="Composite control surface of runtime risk dimensions"
        >
          <RadarSurface points={riskPoints} />
        </ChartCard>
        <div className="glass-panel p-5">
          <div className="section-kicker">Guardrail Drilldown</div>
          {activeGuardrail ? (
            <>
              <h3 className="mt-2 text-lg font-semibold text-white">{activeGuardrail.label}</h3>
              <div className={`mt-2 inline-flex rounded-full border px-3 py-1 text-xs uppercase tracking-[0.2em] ${toneClasses[toneFromStatus(activeGuardrail.status)]}`}>
                {activeGuardrail.status}
              </div>
              <p className="mt-3 text-sm leading-6 text-slate-300/80">{activeGuardrail.detail}</p>
            </>
          ) : (
            <div className="mt-3 text-sm text-slate-400">No guardrail selected.</div>
          )}
        </div>
      </div>
      <div className="glass-panel p-5">
        <div className="section-kicker">Risk Radar</div>
        <h2 className="mt-2 text-xl font-semibold text-white">Constraint-aware readiness surface</h2>
        <p className="mt-3 text-sm leading-6 text-slate-300/75">{snapshot.risk.summary}</p>
        <div className="mt-5 grid gap-4 md:grid-cols-2">
          {snapshot.risk.guardrails.map((guardrail) => (
            <GuardrailCard
              key={guardrail.label}
              guardrail={guardrail}
              active={guardrail.label === selectedGuardrail}
              onClick={() => setSelectedGuardrail(guardrail.label)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function AuditPanel({ snapshot }: { snapshot: TerminalSnapshot }) {
  const [selectedEventKey, setSelectedEventKey] = useState<string | null>(snapshot.audit.events[0]?.timestamp ?? null);
  const eventSeries = useMemo(() => {
    const buckets = new Map<string, number>();
    for (const event of [...snapshot.audit.events].reverse()) {
      const key = event.type || "event";
      buckets.set(key, (buckets.get(key) ?? 0) + 1);
    }
    return [...buckets.entries()].map(([label, count]) => ({ label, value: count }));
  }, [snapshot.audit.events]);
  const pnlSeries = useMemo(
    () => [...snapshot.audit.trades].reverse().slice(0, 40).map((trade) => trade.pnl),
    [snapshot.audit.trades],
  );
  const activeEvent = snapshot.audit.events.find((event) => event.timestamp === selectedEventKey) ?? snapshot.audit.events[0];

  useEffect(() => {
    if (!snapshot.audit.events.length) return;
    if (!selectedEventKey || !snapshot.audit.events.some((event) => event.timestamp === selectedEventKey)) {
      setSelectedEventKey(snapshot.audit.events[0].timestamp);
    }
  }, [selectedEventKey, snapshot.audit.events]);

  return (
    <div className="space-y-5">
      <div className="glass-panel p-5">
        <div className="section-kicker">Research & Audit</div>
        <h2 className="mt-2 text-xl font-semibold text-white">Replayable decision reconstruction</h2>
        <p className="mt-3 text-sm leading-6 text-slate-300/75">{snapshot.audit.summary}</p>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <ChartCard title="Audit Event Mix" subtitle="Distribution of recorded decision events">
          <BarSurface values={eventSeries.map((row) => row.value)} labels={eventSeries.map((row) => row.label)} positiveColor="#52d7ff" negativeColor="#52d7ff" />
        </ChartCard>
        <ChartCard title="Trade PnL Sequence" subtitle="Chronological closed-trade outcomes">
          <BarSurface values={pnlSeries} positiveColor="#2ae6b8" negativeColor="#ff6b88" />
        </ChartCard>
      </div>
      {activeEvent ? (
        <div className="glass-panel p-5">
          <div className="section-kicker">Event Drilldown</div>
          <h3 className="mt-2 text-lg font-semibold text-white">{activeEvent.type}</h3>
          <div className="mt-2 font-mono text-sm text-cyan">{activeEvent.timestamp}</div>
          <p className="mt-3 text-sm leading-6 text-slate-300/80">{activeEvent.detail}</p>
        </div>
      ) : null}
      <div className="grid gap-4 xl:grid-cols-2">
        <TradesPanel trades={snapshot.audit.trades} />
        <EventsPanel events={snapshot.audit.events} selectedEventKey={selectedEventKey} onSelectEvent={setSelectedEventKey} />
      </div>
    </div>
  );
}

function ReasoningTreePanel({
  title,
  subtitle,
  reasoning,
}: {
  title: string;
  subtitle: string;
  reasoning?: ReasoningTree;
}) {
  return (
    <div className="glass-panel p-5">
      <div className="section-kicker">Reasoning Vector</div>
      <h2 className="mt-2 text-xl font-semibold text-white">{title}</h2>
      <p className="mt-3 text-sm leading-6 text-slate-300/75">{subtitle}</p>
      {reasoning && Object.keys(reasoning).length ? (
        <div className="mt-5 space-y-3">
          {Object.entries(reasoning).map(([key, value]) => (
            <ReasoningNodeView key={key} nodeKey={key} value={value} depth={0} />
          ))}
        </div>
      ) : (
        <div className="mt-5 rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-sm text-slate-400">
          No structured reasoning payload is attached to the current signal yet.
        </div>
      )}
    </div>
  );
}

function ReasoningNodeView({
  nodeKey,
  value,
  depth,
}: {
  nodeKey: string;
  value: unknown;
  depth: number;
}) {
  const indentClass = depth === 0 ? "" : depth === 1 ? "ml-4" : depth === 2 ? "ml-8" : "ml-12";

  if (Array.isArray(value)) {
    return (
      <details className={`rounded-2xl border border-white/10 bg-white/[0.03] ${indentClass}`} open={depth < 1}>
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-medium tracking-[0.08em] text-cyan">
          {nodeKey} <span className="text-slate-500">[{value.length}]</span>
        </summary>
        <div className="space-y-2 border-t border-white/8 px-3 py-3">
          {value.map((item, index) => (
            <ReasoningNodeView key={`${nodeKey}-${index}`} nodeKey={`${index}`} value={item} depth={depth + 1} />
          ))}
        </div>
      </details>
    );
  }

  if (value && typeof value === "object") {
    return (
      <details className={`rounded-2xl border border-white/10 bg-white/[0.03] ${indentClass}`} open={depth < 1}>
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-medium tracking-[0.08em] text-cyan">
          {nodeKey}
        </summary>
        <div className="space-y-2 border-t border-white/8 px-3 py-3">
          {Object.entries(value as Record<string, unknown>).map(([childKey, childValue]) => (
            <ReasoningNodeView key={`${nodeKey}-${childKey}`} nodeKey={childKey} value={childValue} depth={depth + 1} />
          ))}
        </div>
      </details>
    );
  }

  return (
    <div className={`rounded-2xl border border-white/8 bg-black/20 px-4 py-3 ${indentClass}`}>
      <div className="flex items-start justify-between gap-4">
        <div className="text-xs uppercase tracking-[0.16em] text-slate-500">{nodeKey}</div>
        <div className="max-w-[70%] text-right text-sm text-white">{formatReasoningValue(value)}</div>
      </div>
    </div>
  );
}

function formatReasoningValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "null";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  return String(value);
}

function RightRail({ snapshot }: { snapshot: TerminalSnapshot }) {
  return (
    <div className="space-y-5">
      <div className="glass-panel p-5">
        <div className="section-kicker">System Telemetry</div>
        <div className="mt-3 grid gap-3">
          <TelemetryRow icon={<ShieldCheck className="h-4 w-4" />} label="Source" value={snapshot.meta.source} />
          <TelemetryRow icon={<Gauge className="h-4 w-4" />} label="Model Version" value={snapshot.meta.modelVersion} />
          <TelemetryRow icon={<Orbit className="h-4 w-4" />} label="Regime" value={snapshot.regime.current} />
          <TelemetryRow icon={<AlertTriangle className="h-4 w-4" />} label="Risk Summary" value={snapshot.risk.summary} />
        </div>
      </div>

      <div className="glass-panel p-5">
        <div className="section-kicker">Top Guardrails</div>
        <div className="mt-4 space-y-3">
          {snapshot.risk.guardrails.slice(0, 3).map((guardrail) => (
            <GuardrailCard key={guardrail.label} guardrail={guardrail} compact />
          ))}
        </div>
      </div>

      <div className="glass-panel overflow-hidden p-5">
        <div className="section-kicker">Signal Stack</div>
        <div className="mt-4 space-y-3">
          {snapshot.signals.candidates.slice(0, 3).map((candidate, index) => (
            <motion.div
              key={candidate.id}
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.06 * index }}
              className="rounded-2xl border border-white/10 bg-white/[0.03] p-4"
            >
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="font-medium text-white">{candidate.asset}</div>
                  <div className="mt-1 text-xs uppercase tracking-[0.2em] text-slate-400">{candidate.tier}</div>
                </div>
                <div className="font-mono text-lg text-cyan">{candidate.confluence.toFixed(2)}</div>
              </div>
              <div className="mt-3 h-2 rounded-full bg-white/8">
                <div className="h-full rounded-full bg-gradient-to-r from-cyan to-teal" style={{ width: `${candidate.confluence * 100}%` }} />
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StatusPill({ label, tone }: { label: string; tone: MetricTile["tone"] }) {
  return <div className={`rounded-full border px-4 py-2 text-xs uppercase tracking-[0.22em] ${toneClasses[tone]}`}>{label}</div>;
}

function MetricCard({ metric }: { metric: MetricTile }) {
  return (
    <motion.div whileHover={{ y: -4 }} className="metric-card">
      <div className={`inline-flex rounded-full border px-3 py-1 text-[0.68rem] uppercase tracking-[0.22em] ${toneClasses[metric.tone]}`}>
        {metric.label}
      </div>
      <div className="mt-4 text-3xl font-semibold tracking-tight text-white">{metric.value}</div>
      {metric.delta ? <div className="mt-2 text-xs uppercase tracking-[0.18em] text-slate-400">{metric.delta}</div> : null}
    </motion.div>
  );
}

function NarrativePanel({ title, text }: { title: string; text: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
      <div className="font-medium text-white">{title}</div>
      <p className="mt-2 text-sm leading-6 text-slate-400">{text}</p>
    </div>
  );
}

function GaugeCard({ label, value }: { label: string; value: number }) {
  const tone = value >= 70 ? "rose" : value >= 40 ? "amber" : "teal";
  return (
    <div className="metric-card">
      <div className="section-kicker">{label}</div>
      <div className="mt-4 flex items-end justify-between">
        <div className={`text-4xl font-semibold ${toneClasses[tone].split(" ")[0]}`}>{value}%</div>
        <Gauge className="h-5 w-5 text-slate-400" />
      </div>
      <div className="mt-4 h-2 rounded-full bg-white/8">
        <div className={`h-full rounded-full ${tone === "teal" ? "bg-gradient-to-r from-teal to-cyan" : tone === "amber" ? "bg-gradient-to-r from-amber to-orange-400" : "bg-gradient-to-r from-rose to-red-500"}`} style={{ width: `${Math.max(6, value)}%` }} />
      </div>
    </div>
  );
}

function GuardrailCard({
  guardrail,
  compact = false,
  active = false,
  onClick,
}: {
  guardrail: Guardrail;
  compact?: boolean;
  active?: boolean;
  onClick?: () => void;
}) {
  const tone = toneClasses[toneFromStatus(guardrail.status)];
  const Tag = onClick ? "button" : "div";
  return (
    <Tag
      onClick={onClick}
      className={`rounded-2xl border p-4 ${tone} ${onClick ? "w-full text-left transition hover:border-cyan/30 hover:bg-cyan/5" : ""} ${active ? "border-cyan/45" : ""}`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="font-medium text-white">{guardrail.label}</div>
        <div className="text-[0.68rem] uppercase tracking-[0.2em]">{guardrail.status}</div>
      </div>
      {!compact ? <p className="mt-2 text-sm leading-6 text-slate-200/85">{guardrail.detail}</p> : <div className="mt-2 text-sm text-slate-300/85">{guardrail.detail}</div>}
    </Tag>
  );
}

function BucketPanel({
  title,
  rows,
}: {
  title: string;
  rows: { label: string; pnl: number; trades: number; winRate: number }[];
}) {
  return (
    <div className="glass-panel p-5">
      <div className="section-kicker">{title}</div>
      <div className="mt-4 space-y-3">
        {rows.length ? rows.map((row) => (
          <div key={`${title}-${row.label}`} className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="font-medium text-white">{row.label}</div>
              <div className={`font-mono ${row.pnl >= 0 ? "text-teal" : "text-rose"}`}>{fmtSignedUsd(row.pnl)}</div>
            </div>
            <div className="mt-2 text-sm text-slate-400">{row.trades} trades • {row.winRate.toFixed(1)}% win rate</div>
          </div>
        )) : (
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-sm text-slate-400">No rows yet.</div>
        )}
      </div>
    </div>
  );
}

function TradesPanel({ trades }: { trades: AuditTrade[] }) {
  return (
    <div className="glass-panel p-5">
      <div className="section-kicker">Trade Ledger</div>
      <div className="mt-4 space-y-3">
        {trades.map((trade) => (
          <div key={trade.tradeId} className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="font-medium text-white">{trade.asset} <span className="text-slate-400">/ {trade.leg}</span></div>
                <div className="mt-1 text-xs uppercase tracking-[0.2em] text-slate-400">{trade.tier}</div>
              </div>
              <div className={`font-mono text-lg ${trade.pnl >= 0 ? "text-teal" : "text-rose"}`}>{trade.pnl >= 0 ? "+" : ""}{trade.pnl.toFixed(0)}</div>
            </div>
            <div className="mt-3 flex items-center justify-between text-sm text-slate-400">
              <span>{trade.reason}</span>
              <span>{trade.r.toFixed(2)}R</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EventsPanel({
  events,
  selectedEventKey,
  onSelectEvent,
}: {
  events: AuditEvent[];
  selectedEventKey?: string | null;
  onSelectEvent?: (eventKey: string) => void;
}) {
  return (
    <div className="glass-panel p-5">
      <div className="section-kicker">Decision Tape</div>
      <div className="mt-4 space-y-3">
        {events.map((event) => (
          <button
            key={`${event.timestamp}-${event.type}`}
            onClick={() => onSelectEvent?.(event.timestamp)}
            className={`w-full rounded-2xl border bg-white/[0.03] p-4 text-left transition hover:border-cyan/30 hover:bg-cyan/5 ${selectedEventKey === event.timestamp ? "border-cyan/40" : "border-white/10"}`}
          >
            <div className="flex items-center justify-between gap-3">
              <div className="font-medium text-white">{event.type}</div>
              <div className="font-mono text-xs text-slate-400">{event.timestamp}</div>
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-300/85">{event.detail}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

function TelemetryRow({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3">
      <div className="rounded-xl border border-cyan/20 bg-cyan/10 p-2 text-cyan">{icon}</div>
      <div className="min-w-0">
        <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{label}</div>
        <div className="truncate text-sm text-white">{value}</div>
      </div>
    </div>
  );
}

function TimeframeMiniMap({
  label,
  candles,
  active,
  focusTime,
  onClick,
}: {
  label: "15m" | "1h" | "6h" | "12h";
  candles: TerminalSnapshot["market"]["candles"];
  active: boolean;
  focusTime: number | null;
  onClick: () => void;
}) {
  const sample = candles.slice(-120);
  if (!sample.length) {
    return (
      <button
        onClick={onClick}
        className={`rounded-2xl border px-4 py-4 text-left ${active ? "border-cyan/45 bg-cyan/10" : "border-white/10 bg-white/[0.03]"}`}
      >
        <div className="section-kicker">{label}</div>
        <div className="mt-2 text-sm text-slate-400">No candles available</div>
      </button>
    );
  }

  const width = 240;
  const height = 64;
  let minClose = sample[0].close;
  let maxClose = sample[0].close;
  for (const row of sample) {
    minClose = Math.min(minClose, row.close);
    maxClose = Math.max(maxClose, row.close);
  }
  const pad = (maxClose - minClose) * 0.08 || Math.max(1, maxClose * 0.002);
  const minY = minClose - pad;
  const maxY = maxClose + pad;
  const span = Math.max(1e-9, maxY - minY);
  const points = sample
    .map((row, index) => {
      const x = (index / (sample.length - 1 || 1)) * width;
      const y = height - ((row.close - minY) / span) * height;
      return `${x},${y}`;
    })
    .join(" ");

  const last = sample[sample.length - 1];
  const first = sample[0];
  const change = first.close ? ((last.close / first.close) - 1) * 100 : 0;
  const trendUp = change >= 0;
  const focusIndex = (() => {
    if (focusTime === null) return sample.length - 1;
    for (let i = 0; i < sample.length; i += 1) {
      const next = sample[i + 1];
      if (sample[i].time <= focusTime && (!next || next.time > focusTime)) {
        return i;
      }
    }
    return sample.length - 1;
  })();
  const focusX = (focusIndex / (sample.length - 1 || 1)) * width;
  const focusClose = sample[focusIndex]?.close ?? last.close;

  return (
    <button
      onClick={onClick}
      className={`rounded-2xl border p-3 text-left transition ${active ? "border-cyan/45 bg-cyan/10 shadow-bloom" : "border-white/10 bg-white/[0.03] hover:border-cyan/35 hover:bg-cyan/5"}`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="section-kicker">{label}</div>
        <div className={`font-mono text-xs ${trendUp ? "text-teal" : "text-rose"}`}>{change >= 0 ? "+" : ""}{change.toFixed(2)}%</div>
      </div>
      <div className="mt-2 overflow-hidden rounded-xl border border-white/10 bg-[#06111d]/80 p-2">
        <svg width="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="h-[60px] w-full">
          <polyline
            fill="none"
            stroke={trendUp ? "#2ae6b8" : "#ff6b88"}
            strokeWidth="2.2"
            points={points}
            vectorEffect="non-scaling-stroke"
          />
          <line x1={focusX} y1={0} x2={focusX} y2={height} stroke="rgba(82,215,255,0.42)" strokeDasharray="4 4" />
        </svg>
      </div>
      <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
        <span>{fmtUsd(last.close)}</span>
        <span className="text-cyan">{fmtUsd(focusClose)}</span>
      </div>
    </button>
  );
}

function ChartCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
}) {
  return (
    <div className="glass-panel p-5">
      <div className="section-kicker">{title}</div>
      <div className="mt-2 text-sm text-slate-300/75">{subtitle}</div>
      <div className="mt-4 overflow-hidden rounded-2xl border border-white/10 bg-[#06111d]/70 p-3">
        {children}
      </div>
    </div>
  );
}

function InteractiveLineSurface({
  values,
  labels,
  color,
  fill,
  valueFormatter,
}: {
  values: number[];
  labels?: string[];
  color: string;
  fill?: string;
  valueFormatter?: (value: number) => string;
}) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  if (!values.length) {
    return <div className="h-[190px] text-sm text-slate-400">No series yet.</div>;
  }

  const width = 640;
  const height = 190;
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const pad = (maxValue - minValue) * 0.08 || Math.max(Math.abs(maxValue) * 0.02, 1);
  const minY = minValue - pad;
  const maxY = maxValue + pad;
  const span = Math.max(1e-9, maxY - minY);
  const points = values
    .map((value, idx) => {
      const x = (idx / (values.length - 1 || 1)) * width;
      const y = height - ((value - minY) / span) * height;
      return `${x},${y}`;
    })
    .join(" ");
  const area = `0,${height} ${points} ${width},${height}`;
  const activeIndex = hoverIndex ?? (values.length - 1);
  const activeValue = values[activeIndex] ?? values[values.length - 1];
  const activeLabel = labels?.[activeIndex] ?? `#${activeIndex + 1}`;
  const activeX = (activeIndex / (values.length - 1 || 1)) * width;
  const activeY = height - ((activeValue - minY) / span) * height;
  const tooltipLeft = `${(activeIndex / (values.length - 1 || 1)) * 100}%`;
  const gradientId = `surface-interactive-${color.replace(/[^a-zA-Z0-9]/g, "")}`;
  const zeroY = height - ((0 - minY) / span) * height;

  return (
    <div className="relative">
      <div
        className="pointer-events-none absolute top-2 z-20 -translate-x-1/2 rounded-lg border border-white/15 bg-[#06111d]/95 px-2 py-1 text-xs text-slate-200 shadow-[0_8px_24px_rgba(0,0,0,0.45)]"
        style={{ left: tooltipLeft }}
      >
        <div className="font-mono text-cyan">{valueFormatter ? valueFormatter(activeValue) : activeValue.toFixed(3)}</div>
        <div className="text-[10px] uppercase tracking-[0.16em] text-slate-400">{activeLabel}</div>
      </div>
      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="h-[190px] w-full"
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const ratio = (event.clientX - rect.left) / Math.max(rect.width, 1);
          const clamped = Math.max(0, Math.min(1, ratio));
          setHoverIndex(Math.round(clamped * (values.length - 1)));
        }}
        onMouseLeave={() => setHoverIndex(null)}
      >
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={fill ?? `${color}66`} />
            <stop offset="100%" stopColor="rgba(0,0,0,0)" />
          </linearGradient>
        </defs>
        <polyline fill={`url(#${gradientId})`} points={area} />
        <polyline fill="none" stroke={color} strokeWidth="2.4" points={points} vectorEffect="non-scaling-stroke" />
        <line x1={0} y1={zeroY} x2={width} y2={zeroY} stroke="rgba(148,163,184,0.22)" strokeDasharray="5 4" />
        <line x1={activeX} y1={0} x2={activeX} y2={height} stroke="rgba(82,215,255,0.42)" strokeDasharray="4 3" />
        <circle cx={activeX} cy={activeY} r={4.2} fill={color} stroke="rgba(255,255,255,0.85)" strokeWidth="1.1" />
      </svg>
    </div>
  );
}

function InteractiveBarSurface({
  values,
  labels,
  positiveColor,
  negativeColor,
  valueFormatter,
}: {
  values: number[];
  labels?: string[];
  positiveColor: string;
  negativeColor: string;
  valueFormatter?: (value: number) => string;
}) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  if (!values.length) {
    return <div className="h-[190px] text-sm text-slate-400">No bars yet.</div>;
  }
  const width = 640;
  const height = 190;
  const maxAbs = Math.max(...values.map((value) => Math.abs(value)), 1);
  const barWidth = width / values.length;
  const zeroY = height / 2;
  const activeIndex = hoverIndex ?? (values.length - 1);
  const activeValue = values[activeIndex] ?? values[values.length - 1];
  const activeLabel = labels?.[activeIndex] ?? `#${activeIndex + 1}`;
  const tooltipLeft = `${((activeIndex + 0.5) / values.length) * 100}%`;

  return (
    <div className="relative">
      <div
        className="pointer-events-none absolute top-2 z-20 -translate-x-1/2 rounded-lg border border-white/15 bg-[#06111d]/95 px-2 py-1 text-xs text-slate-200 shadow-[0_8px_24px_rgba(0,0,0,0.45)]"
        style={{ left: tooltipLeft }}
      >
        <div className={activeValue >= 0 ? "font-mono text-teal" : "font-mono text-rose"}>
          {valueFormatter ? valueFormatter(activeValue) : activeValue.toFixed(3)}
        </div>
        <div className="text-[10px] uppercase tracking-[0.16em] text-slate-400">{activeLabel}</div>
      </div>
      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="h-[190px] w-full"
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const ratio = (event.clientX - rect.left) / Math.max(rect.width, 1);
          const clamped = Math.max(0, Math.min(1, ratio));
          setHoverIndex(Math.min(values.length - 1, Math.floor(clamped * values.length)));
        }}
        onMouseLeave={() => setHoverIndex(null)}
      >
        <line x1={0} y1={zeroY} x2={width} y2={zeroY} stroke="rgba(148,163,184,0.28)" strokeDasharray="4 3" />
        {values.map((value, idx) => {
          const barHeight = (Math.abs(value) / maxAbs) * (height * 0.42);
          const x = idx * barWidth + barWidth * 0.14;
          const y = value >= 0 ? zeroY - barHeight : zeroY;
          const isActive = idx === activeIndex;
          return (
            <rect
              key={`${idx}-${value}`}
              x={x}
              y={y}
              width={Math.max(2, barWidth * 0.72)}
              height={Math.max(2, barHeight)}
              fill={value >= 0 ? positiveColor : negativeColor}
              opacity={isActive ? 1 : 0.84}
              rx={2}
              stroke={isActive ? "rgba(255,255,255,0.75)" : "none"}
              strokeWidth={isActive ? 1 : 0}
            />
          );
        })}
      </svg>
    </div>
  );
}

function LineSurface({
  values,
  color,
  fill,
}: {
  values: number[];
  color: string;
  fill?: string;
}) {
  if (!values.length) {
    return <div className="h-[190px] text-sm text-slate-400">No series yet.</div>;
  }
  const width = 640;
  const height = 190;
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const pad = (maxValue - minValue) * 0.08 || Math.max(Math.abs(maxValue) * 0.02, 1);
  const minY = minValue - pad;
  const maxY = maxValue + pad;
  const span = Math.max(1e-9, maxY - minY);
  const points = values
    .map((value, idx) => {
      const x = (idx / (values.length - 1 || 1)) * width;
      const y = height - ((value - minY) / span) * height;
      return `${x},${y}`;
    })
    .join(" ");
  const area = `0,${height} ${points} ${width},${height}`;

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="h-[190px] w-full">
      <defs>
        <linearGradient id={`surface-${color.replace(/[^a-zA-Z0-9]/g, "")}`} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor={fill ?? `${color}66`} />
          <stop offset="100%" stopColor="rgba(0,0,0,0)" />
        </linearGradient>
      </defs>
      <polyline fill={`url(#surface-${color.replace(/[^a-zA-Z0-9]/g, "")})`} points={area} />
      <polyline fill="none" stroke={color} strokeWidth="2.4" points={points} vectorEffect="non-scaling-stroke" />
      <line x1={0} y1={height - ((0 - minY) / span) * height} x2={width} y2={height - ((0 - minY) / span) * height} stroke="rgba(148,163,184,0.22)" strokeDasharray="5 4" />
    </svg>
  );
}

function BarSurface({
  values,
  labels,
  positiveColor,
  negativeColor,
}: {
  values: number[];
  labels?: string[];
  positiveColor: string;
  negativeColor: string;
}) {
  if (!values.length) {
    return <div className="h-[190px] text-sm text-slate-400">No bars yet.</div>;
  }
  const width = 640;
  const height = 190;
  const maxAbs = Math.max(...values.map((value) => Math.abs(value)), 1);
  const barWidth = width / values.length;
  const zeroY = height / 2;

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="h-[190px] w-full">
      <line x1={0} y1={zeroY} x2={width} y2={zeroY} stroke="rgba(148,163,184,0.28)" strokeDasharray="4 3" />
      {values.map((value, idx) => {
        const barHeight = (Math.abs(value) / maxAbs) * (height * 0.42);
        const x = idx * barWidth + barWidth * 0.14;
        const y = value >= 0 ? zeroY - barHeight : zeroY;
        const label = labels?.[idx];
        return (
          <g key={`${idx}-${value}`}>
            <rect
              x={x}
              y={y}
              width={Math.max(2, barWidth * 0.72)}
              height={Math.max(2, barHeight)}
              fill={value >= 0 ? positiveColor : negativeColor}
              opacity={0.85}
              rx={2}
            />
            {label ? (
              <text x={x + barWidth * 0.36} y={height - 4} textAnchor="middle" fill="rgba(148,163,184,0.9)" fontSize="10">
                {label}
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

function ScatterSurface({
  points,
}: {
  points: Array<{ x: number; y: number; label: string; tone: "teal" | "rose" }>;
}) {
  const width = 640;
  const height = 190;
  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="h-[190px] w-full">
      <line x1={0} y1={height - 40} x2={width} y2={height - 40} stroke="rgba(148,163,184,0.26)" />
      <line x1={40} y1={0} x2={40} y2={height} stroke="rgba(148,163,184,0.26)" />
      <text x={48} y={16} fill="rgba(148,163,184,0.88)" fontSize="10">Hazard ↑</text>
      <text x={width - 8} y={height - 10} textAnchor="end" fill="rgba(148,163,184,0.88)" fontSize="10">Confluence →</text>
      {points.map((point) => {
        const x = 40 + (Math.max(0, Math.min(100, point.x)) / 100) * (width - 56);
        const y = (height - 40) - (Math.max(0, Math.min(100, point.y)) / 100) * (height - 56);
        const color = point.tone === "teal" ? "#2ae6b8" : "#ff6b88";
        return (
          <g key={`${point.label}-${point.x}-${point.y}`}>
            <circle cx={x} cy={y} r={5.5} fill={color} opacity={0.9} />
            <text x={x + 7} y={y - 6} fill="rgba(222,231,246,0.9)" fontSize="10">{point.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function RadarSurface({
  points,
}: {
  points: Array<{ axis: string; value: number }>;
}) {
  if (!points.length) {
    return <div className="h-[190px] text-sm text-slate-400">No risk vectors yet.</div>;
  }
  const size = 210;
  const center = size / 2;
  const radius = 82;
  const angleStep = (Math.PI * 2) / points.length;
  const polygon = points
    .map((point, idx) => {
      const angle = -Math.PI / 2 + idx * angleStep;
      const r = radius * (Math.max(0, Math.min(100, point.value)) / 100);
      const x = center + Math.cos(angle) * r;
      const y = center + Math.sin(angle) * r;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="flex items-center justify-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="h-[190px] w-[210px]">
        {[0.25, 0.5, 0.75, 1].map((step) => (
          <circle key={step} cx={center} cy={center} r={radius * step} fill="none" stroke="rgba(148,163,184,0.2)" />
        ))}
        {points.map((point, idx) => {
          const angle = -Math.PI / 2 + idx * angleStep;
          const x = center + Math.cos(angle) * (radius + 16);
          const y = center + Math.sin(angle) * (radius + 16);
          const axisX = center + Math.cos(angle) * radius;
          const axisY = center + Math.sin(angle) * radius;
          return (
            <g key={point.axis}>
              <line x1={center} y1={center} x2={axisX} y2={axisY} stroke="rgba(148,163,184,0.22)" />
              <text x={x} y={y} textAnchor="middle" fill="rgba(222,231,246,0.86)" fontSize="10">{point.axis}</text>
            </g>
          );
        })}
        <polygon points={polygon} fill="rgba(82,215,255,0.22)" stroke="#52d7ff" strokeWidth="2" />
      </svg>
    </div>
  );
}

function DrillValue({
  label,
  value,
  tone = "slate",
}: {
  label: string;
  value: string;
  tone?: MetricTile["tone"];
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3">
      <div className="text-[0.64rem] uppercase tracking-[0.18em] text-slate-500">{label}</div>
      <div className={`mt-2 text-sm ${toneClasses[tone].split(" ")[0]}`}>{value}</div>
    </div>
  );
}

function tradeTsMs(trade: AuditTrade): number {
  const parsed = Date.parse(trade.exitTs || trade.entryTs || "");
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function fmtUsd(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function fmtSignedUsd(value: number): string {
  const abs = fmtUsd(Math.abs(value));
  if (value > 0) return `+${abs}`;
  if (value < 0) return `-${abs}`;
  return abs;
}

function fmtRatioPct(value?: number | null): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function fmtMaybeNumber(value?: number | null, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

function fmtTs(value: string): string {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function fmtUnix(value: number): string {
  const ts = value > 10_000_000_000 ? value : value * 1000;
  const parsed = new Date(ts);
  if (Number.isNaN(parsed.getTime())) return "-";
  return parsed.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function toneFromStatus(status: Guardrail["status"]): MetricTile["tone"] {
  if (status === "pass") return "teal";
  if (status === "warn") return "amber";
  return "rose";
}
