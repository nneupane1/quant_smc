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

import type { AuditEvent, AuditTrade, Guardrail, InsightNode, MetricTile, ReasoningTree, SignalCandidate, TerminalSnapshot } from "@/lib/terminal-types";
import { MarketCanvas } from "@/components/market-canvas";

const DOMAINS = [
  { id: "mission", label: "Mission Control", caption: "Execution desk, cycle capital, open risk", icon: Activity },
  { id: "market", label: "Market Canvas", caption: "TradingView-style candles + quant overlays", icon: CandlestickChart },
  { id: "performance", label: "Performance Intel", caption: "PnL, RR, win-rate and trade ledger", icon: BarChart3 },
  { id: "insights", label: "Insights", caption: "Causal trace, structure, eligibility", icon: BrainCircuit },
  { id: "regime", label: "Regime Briefings", caption: "12h state, persistence, transition risk", icon: Orbit },
  { id: "signals", label: "Signal Intelligence", caption: "Ranked candidates, confluence, coherence", icon: AudioWaveform },
  { id: "risk", label: "Risk Radar", caption: "Stress, slippage, exposure, gates", icon: Radar },
  { id: "audit", label: "Research & Audit", caption: "Traceability, trades, replayable events", icon: DatabaseZap },
] as const;

type DomainId = (typeof DOMAINS)[number]["id"];

const toneClasses: Record<MetricTile["tone"], string> = {
  cyan: "text-cyan border-cyan/20 bg-cyan/10",
  teal: "text-teal border-teal/20 bg-teal/10",
  amber: "text-amber border-amber/20 bg-amber/10",
  rose: "text-rose border-rose/20 bg-rose/10",
  slate: "text-slate-200 border-white/10 bg-white/5",
};

function normalizeSnapshot(snapshot: TerminalSnapshot): TerminalSnapshot {
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
  if (snapshot.performance && snapshot.market) return { ...snapshot, market: normalizedMarket };
  return {
    ...snapshot,
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

export function TerminalApp({ initialSnapshot }: { initialSnapshot: TerminalSnapshot }) {
  const [activeDomain, setActiveDomain] = useState<DomainId>("mission");
  const [hoveredDomain, setHoveredDomain] = useState<DomainId | null>(null);
  const [snapshot, setSnapshot] = useState(normalizeSnapshot(initialSnapshot));
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
        const res = await fetch("/api/terminal", { cache: "no-store" });
        if (!res.ok || cancelled) return;
        const next = (await res.json()) as TerminalSnapshot;
        if (!cancelled) {
          startTransition(() => setSnapshot(normalizeSnapshot(next)));
        }
      } catch {
        // keep last successful state
      }
    };

    if (!wsConnected) {
      refreshSnapshot();
    }
    const timer = window.setInterval(() => {
      if (!wsConnected) {
        refreshSnapshot();
      }
    }, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [wsConnected]);

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
  }, [wsUrl]);

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
    <div className="glass-panel overflow-hidden px-5 py-4">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan/70 to-transparent" />
      <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
        <div className="flex items-start gap-6">
          <div className="relative hidden h-[148px] w-[292px] shrink-0 overflow-hidden rounded-[32px] border border-cyan/35 bg-[#04101b] shadow-bloom lg:block lg:translate-y-5">
            <Image
              src="/bull_bear.png"
              alt="Quant SMC Bull Bear"
              fill
              priority
              className="object-cover object-center scale-[1.08] [clip-path:inset(1.2%_1.2%_1.2%_1.2%_round_34px)]"
            />
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-tr from-cyan/20 via-transparent to-amber/25" />
            <div className="pointer-events-none absolute inset-0 rounded-[32px] ring-1 ring-inset ring-cyan/20" />
          </div>
          <div>
            <div className="section-kicker">Live Trading Room</div>
            <div className="mt-2 font-[var(--font-display)] text-3xl font-semibold tracking-tight text-white md:text-4xl xl:whitespace-nowrap">
              Deterministic Execution Terminal
            </div>
            <p className="mt-2 max-w-3xl text-xs uppercase tracking-[0.18em] text-cyan/75">{snapshot.mission.headline}</p>
            <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-300/75">{snapshot.mission.substatus}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-3">
          <StatusPill label={snapshot.meta.source === "artifacts" ? "Artifact parity live" : "Demo parity preview"} tone={snapshot.meta.source === "artifacts" ? "teal" : "amber"} />
          <StatusPill label={wsConnected ? "WS linked" : "WS standby"} tone={wsConnected ? "teal" : "amber"} />
          <StatusPill label={`Models ${snapshot.meta.modelVersion}`} tone="cyan" />
          <StatusPill label={snapshot.mission.status} tone={snapshot.mission.status === "Cooling" ? "rose" : "teal"} />
        </div>
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
  const tableRows = snapshot.performance.tradeTable.slice(0, 30);
  const [selectedTradeId, setSelectedTradeId] = useState<string | null>(tableRows[0]?.tradeId ?? null);
  const equityCurve = useMemo(() => {
    const chronological = [...tableRows].reverse();
    let running = 20_000;
    return chronological.map((trade) => {
      running += trade.pnl;
      return running;
    });
  }, [tableRows]);
  const rSeries = useMemo(() => [...tableRows].reverse().map((trade) => trade.r), [tableRows]);
  const selectedTrade = tableRows.find((trade) => trade.tradeId === selectedTradeId) ?? tableRows[0];

  useEffect(() => {
    if (!tableRows.length) {
      setSelectedTradeId(null);
      return;
    }
    if (!selectedTradeId || !tableRows.some((trade) => trade.tradeId === selectedTradeId)) {
      setSelectedTradeId(tableRows[0].tradeId);
    }
  }, [selectedTradeId, tableRows]);

  return (
    <div className="space-y-5">
      <div className="glass-panel p-5">
        <div className="section-kicker">Performance Intelligence</div>
        <h2 className="mt-2 text-xl font-semibold text-white">PnL, risk-to-reward, and execution quality in one control surface</h2>
        <p className="mt-3 text-sm leading-6 text-slate-300/75">{snapshot.performance.summary}</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {snapshot.performance.kpis.map((metric) => (
          <MetricCard key={metric.label} metric={metric} />
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <ChartCard
          title="Equity Curve"
          subtitle="Simulated equity trajectory from closed trades"
        >
          <LineSurface values={equityCurve} color="#52d7ff" fill="rgba(82, 215, 255, 0.14)" />
        </ChartCard>
        <ChartCard
          title="R-Multiple Distribution"
          subtitle="Chronological trade R outcomes"
        >
          <BarSurface values={rSeries} positiveColor="#2ae6b8" negativeColor="#ff6b88" />
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

      <div className="grid gap-4 lg:grid-cols-2">
        <BucketPanel title="By Asset" rows={snapshot.performance.byAsset} />
        <BucketPanel title="By Tier" rows={snapshot.performance.byTier} />
      </div>

      <div className="glass-panel overflow-hidden">
        <div className="border-b border-white/10 px-5 py-4">
          <div className="section-kicker">Trade Table</div>
          <div className="mt-2 text-sm text-slate-300">Full trade details for PnL attribution, R-multiple behavior, and execution quality.</div>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-white/10 bg-white/[0.03] text-xs uppercase tracking-[0.18em] text-slate-400">
              <tr>
                {["Trade", "Asset", "Side", "Tier", "PnL", "R", "Qty", "Notional", "Entry", "Exit", "Hold", "Fees", "Slip", "Reason"].map((header) => (
                  <th key={header} className="px-3 py-3 font-medium">{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((trade) => (
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
                  <td className={`px-3 py-3 font-medium ${trade.pnl >= 0 ? "text-teal" : "text-rose"}`}>{fmtSignedUsd(trade.pnl)}</td>
                  <td className="px-3 py-3 text-cyan">{trade.r.toFixed(2)}R</td>
                  <td className="px-3 py-3 text-slate-200">{trade.qty ? trade.qty.toFixed(4) : "-"}</td>
                  <td className="px-3 py-3 text-slate-200">{trade.notional ? fmtUsd(trade.notional) : "-"}</td>
                  <td className="px-3 py-3 text-slate-400">{fmtTs(trade.entryTs)}</td>
                  <td className="px-3 py-3 text-slate-400">{fmtTs(trade.exitTs || "")}</td>
                  <td className="px-3 py-3 text-slate-300">{trade.holdMinutes ? `${Math.round(trade.holdMinutes)}m` : "-"}</td>
                  <td className="px-3 py-3 text-amber">{trade.fees ? fmtUsd(trade.fees) : "-"}</td>
                  <td className="px-3 py-3 text-slate-300">{trade.slippageBps ? `${trade.slippageBps.toFixed(2)} bps` : "-"}</td>
                  <td className="px-3 py-3 text-slate-400">{trade.reason}</td>
                </tr>
              ))}
              {tableRows.length === 0 ? (
                <tr>
                  <td colSpan={14} className="px-4 py-6 text-center text-slate-400">No trade rows available yet.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>

      {selectedTrade ? (
        <div className="glass-panel p-5">
          <div className="section-kicker">Selected Trade Drilldown</div>
          <div className="mt-3 grid gap-3 md:grid-cols-4">
            <DrillValue label="Trade" value={selectedTrade.tradeId} />
            <DrillValue label="Asset" value={selectedTrade.asset} />
            <DrillValue label="PnL" value={fmtSignedUsd(selectedTrade.pnl)} tone={selectedTrade.pnl >= 0 ? "teal" : "rose"} />
            <DrillValue label="R Multiple" value={`${selectedTrade.r.toFixed(2)}R`} tone="cyan" />
            <DrillValue label="Entry" value={fmtTs(selectedTrade.entryTs)} />
            <DrillValue label="Exit" value={fmtTs(selectedTrade.exitTs || "")} />
            <DrillValue label="Notional" value={selectedTrade.notional ? fmtUsd(selectedTrade.notional) : "-"} />
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
