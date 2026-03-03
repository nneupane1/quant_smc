"use client";

import { startTransition, useDeferredValue, useEffect, useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  AudioWaveform,
  BrainCircuit,
  DatabaseZap,
  Gauge,
  Orbit,
  Radar,
  ShieldCheck,
} from "lucide-react";

import type { AuditEvent, AuditTrade, Guardrail, InsightNode, MetricTile, ReasoningTree, SignalCandidate, TerminalSnapshot } from "@/lib/terminal-types";

const DOMAINS = [
  { id: "mission", label: "Mission Control", caption: "Execution desk, cycle capital, open risk", icon: Activity },
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

export function TerminalApp({ initialSnapshot }: { initialSnapshot: TerminalSnapshot }) {
  const [activeDomain, setActiveDomain] = useState<DomainId>("mission");
  const [hoveredDomain, setHoveredDomain] = useState<DomainId | null>(null);
  const [snapshot, setSnapshot] = useState(initialSnapshot);
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
          startTransition(() => setSnapshot(next));
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
            startTransition(() => setSnapshot(payload.data as TerminalSnapshot));
          } else if (payload.type === "bootstrap" && payload.data && "snapshot" in payload.data) {
            startTransition(() => setSnapshot((payload.data as { snapshot: TerminalSnapshot }).snapshot));
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
          <div className="mt-5 grid gap-5 xl:grid-cols-[1.5fr_0.95fr]">
            <AnimatePresence mode="wait">
              <motion.section
                key={activeDomain}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.24, ease: "easeOut" }}
                className="min-w-0"
              >
                {activeDomain === "mission" && <MissionPanel snapshot={snapshot} />}
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

            <RightRail snapshot={snapshot} />
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
        <div>
          <div className="section-kicker">Live Trading Room</div>
          <div className="mt-2 font-[var(--font-display)] text-3xl font-semibold tracking-tight text-white md:text-4xl">
            {snapshot.mission.headline}
          </div>
          <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-300/75">{snapshot.mission.substatus}</p>
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

function InsightsPanel({ summary, trace, reasoning }: { summary: string; trace: InsightNode[]; reasoning?: ReasoningTree }) {
  return (
    <div className="space-y-5">
      <div className="glass-panel p-5">
        <div className="section-kicker">Insights</div>
        <h2 className="mt-2 text-xl font-semibold text-white">State inspection, not indicator clutter</h2>
        <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-300/75">{summary}</p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {trace.map((node) => (
          <motion.div key={node.label} whileHover={{ y: -4 }} className="metric-card p-5">
            <div className={`inline-flex rounded-full border px-3 py-1 text-xs uppercase tracking-[0.2em] ${toneClasses[node.tone]}`}>
              {node.label}
            </div>
            <div className="mt-4 text-2xl font-semibold text-white">{node.value}</div>
            <p className="mt-3 text-sm leading-6 text-slate-300/80">{node.detail}</p>
          </motion.div>
        ))}
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
  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard metric={{ label: "Current Regime", value: snapshot.regime.current, tone: "cyan", delta: "12h anchor" }} />
        <MetricCard metric={{ label: "Persistence", value: `${snapshot.regime.persistence}%`, tone: "teal", delta: "stability" }} />
        <MetricCard metric={{ label: "Transition Risk", value: `${snapshot.regime.transitionRisk}%`, tone: "amber", delta: "state change risk" }} />
      </div>
      <div className="glass-panel p-5">
        <div className="section-kicker">Regime Briefings</div>
        <h2 className="mt-2 text-xl font-semibold text-white">State probabilities with institutional readability</h2>
        <div className="mt-5 space-y-4">
          {snapshot.regime.states.map((state) => (
            <div key={state.name} className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
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
            </div>
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
  return (
    <div className="space-y-5">
      <div className="glass-panel p-5">
        <div className="section-kicker">Signal Intelligence</div>
        <h2 className="mt-2 text-xl font-semibold text-white">Comparative ranking across the active opportunity surface</h2>
        <p className="mt-3 text-sm leading-6 text-slate-300/75">{summary}</p>
      </div>
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
  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-3">
        <GaugeCard label="Stress" value={snapshot.risk.stress} />
        <GaugeCard label="Slippage" value={snapshot.risk.slippage} />
        <GaugeCard label="Exposure" value={snapshot.risk.exposure} />
      </div>
      <div className="glass-panel p-5">
        <div className="section-kicker">Risk Radar</div>
        <h2 className="mt-2 text-xl font-semibold text-white">Constraint-aware readiness surface</h2>
        <p className="mt-3 text-sm leading-6 text-slate-300/75">{snapshot.risk.summary}</p>
        <div className="mt-5 grid gap-4 md:grid-cols-2">
          {snapshot.risk.guardrails.map((guardrail) => (
            <GuardrailCard key={guardrail.label} guardrail={guardrail} />
          ))}
        </div>
      </div>
    </div>
  );
}

function AuditPanel({ snapshot }: { snapshot: TerminalSnapshot }) {
  return (
    <div className="space-y-5">
      <div className="glass-panel p-5">
        <div className="section-kicker">Research & Audit</div>
        <h2 className="mt-2 text-xl font-semibold text-white">Replayable decision reconstruction</h2>
        <p className="mt-3 text-sm leading-6 text-slate-300/75">{snapshot.audit.summary}</p>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <TradesPanel trades={snapshot.audit.trades} />
        <EventsPanel events={snapshot.audit.events} />
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

function GuardrailCard({ guardrail, compact = false }: { guardrail: Guardrail; compact?: boolean }) {
  const tone = toneClasses[toneFromStatus(guardrail.status)];
  return (
    <div className={`rounded-2xl border p-4 ${tone}`}>
      <div className="flex items-center justify-between gap-3">
        <div className="font-medium text-white">{guardrail.label}</div>
        <div className="text-[0.68rem] uppercase tracking-[0.2em]">{guardrail.status}</div>
      </div>
      {!compact ? <p className="mt-2 text-sm leading-6 text-slate-200/85">{guardrail.detail}</p> : <div className="mt-2 text-sm text-slate-300/85">{guardrail.detail}</div>}
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

function EventsPanel({ events }: { events: AuditEvent[] }) {
  return (
    <div className="glass-panel p-5">
      <div className="section-kicker">Decision Tape</div>
      <div className="mt-4 space-y-3">
        {events.map((event) => (
          <div key={`${event.timestamp}-${event.type}`} className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="font-medium text-white">{event.type}</div>
              <div className="font-mono text-xs text-slate-400">{event.timestamp}</div>
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-300/85">{event.detail}</p>
          </div>
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

function toneFromStatus(status: Guardrail["status"]): MetricTile["tone"] {
  if (status === "pass") return "teal";
  if (status === "warn") return "amber";
  return "rose";
}
