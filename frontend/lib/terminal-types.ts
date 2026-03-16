export type DataSource = "artifacts" | "demo" | "telemetry";
export type TerminalMode = "auto" | "backtest" | "forward" | "live";

export type ReasoningValue =
  | string
  | number
  | boolean
  | null
  | ReasoningTree
  | ReasoningValue[];

export type ReasoningTree = {
  [key: string]: ReasoningValue;
};

export type MetricTile = {
  label: string;
  value: string;
  tone: "cyan" | "teal" | "amber" | "rose" | "slate";
  delta?: string;
};

export type Guardrail = {
  label: string;
  status: "pass" | "warn" | "block";
  detail: string;
};

export type InsightNode = {
  label: string;
  value: string;
  detail: string;
  tone: "cyan" | "teal" | "amber" | "rose";
};

export type RegimeState = {
  name: string;
  probability: number;
  description: string;
};

export type SignalCandidate = {
  id: string;
  asset: string;
  side: "long" | "short";
  tier: string;
  confluence: number;
  evr: number;
  flow1h: number;
  hazard: number;
  regime: string;
  reason: string;
  status?: "executable" | "rejected";
  eventType?: string;
  reasoning?: ReasoningTree;
};

export type AuditTrade = {
  tradeId: string;
  asset: string;
  side?: "long" | "short";
  leg: string;
  tier: string;
  pnl: number;
  r: number;
  session?: string;
  regime?: string;
  entryPrice?: number;
  exitPrice?: number;
  qty?: number;
  notional?: number;
  riskUsd?: number;
  fees?: number;
  slippageBps?: number;
  mae?: number;
  mfe?: number;
  holdMinutes?: number;
  status?: "open" | "closed";
  model?: string;
  reason: string;
  entryTs: string;
  exitTs?: string;
};

export type AuditEvent = {
  timestamp: string;
  type: string;
  detail: string;
};

export type PerformanceKpi = {
  label: string;
  value: string;
  tone: MetricTile["tone"];
  delta?: string;
};

export type PerformancePeriod = {
  label: string;
  pnl: number;
  trades: number;
  winRate: number;
  avgR: number;
};

export type PerformanceBucket = {
  label: string;
  pnl: number;
  trades: number;
  winRate: number;
};

export type PerformanceTimelinePoint = {
  label: string;
  ts?: string;
  pnl: number;
  trades: number;
  winRate?: number;
  avgR?: number;
  equity?: number;
  drawdown?: number;
};

export type PerformanceExpectancy = {
  expectancyR: number;
  avgWin: number;
  avgLoss: number;
  payoffRatio: number;
  medianPnl: number;
  medianR: number;
  maxConsecutiveWins: number;
  maxConsecutiveLosses: number;
  maxDrawdown: number;
};

export type MarketCandle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type MarketMarker = {
  time: number;
  position: "aboveBar" | "belowBar" | "inBar";
  color: string;
  shape: "arrowUp" | "arrowDown" | "circle" | "square";
  text: string;
};

export type MarketStat = {
  label: string;
  value: string;
  tone: MetricTile["tone"];
  detail?: string;
};

export type MarketZone = {
  kind: "ob" | "fvg" | "liquidity";
  side: "bullish" | "bearish" | "neutral";
  start: number;
  end: number;
  top: number;
  bottom: number;
  label: string;
  score?: number;
};

export type MarketTimeframes = {
  m15: MarketCandle[];
  h1: MarketCandle[];
  h6: MarketCandle[];
  h12: MarketCandle[];
};

export type MarketSnapshot = {
  symbol: string;
  timeframe: string;
  summary: string;
  candles: MarketCandle[];
  markers: MarketMarker[];
  zones: MarketZone[];
  timeframes: MarketTimeframes;
  stats: MarketStat[];
  activeTrades: AuditTrade[];
};

export type TerminalSnapshot = {
  meta: {
    source: DataSource;
    lastUpdated: string;
    repoRoot: string;
    modelVersion: string;
    transport: string;
    viewModeRequested: TerminalMode;
    viewModeEffective: TerminalMode;
  };
  mission: {
    headline: string;
    status: string;
    substatus: string;
    metrics: MetricTile[];
  };
  insights: {
    summary: string;
    trace: InsightNode[];
    latestReasoning?: ReasoningTree;
  };
  regime: {
    current: string;
    persistence: number;
    transitionRisk: number;
    states: RegimeState[];
  };
  signals: {
    summary: string;
    candidates: SignalCandidate[];
  };
  risk: {
    summary: string;
    stress: number;
    slippage: number;
    exposure: number;
    guardrails: Guardrail[];
  };
  performance: {
    summary: string;
    kpis: PerformanceKpi[];
    periods: PerformancePeriod[];
    byAsset: PerformanceBucket[];
    byTier: PerformanceBucket[];
    byModel?: PerformanceBucket[];
    bySession?: PerformanceBucket[];
    byRegime?: PerformanceBucket[];
    byWeekday?: PerformanceBucket[];
    byHour?: PerformanceBucket[];
    byHold?: PerformanceBucket[];
    topWinners?: AuditTrade[];
    topLosers?: AuditTrade[];
    expectancy?: PerformanceExpectancy;
    timeline?: {
      equity?: PerformanceTimelinePoint[];
      daily?: PerformanceTimelinePoint[];
      monthly?: PerformanceTimelinePoint[];
    };
    tradeTable: AuditTrade[];
  };
  market: MarketSnapshot;
  audit: {
    summary: string;
    trades: AuditTrade[];
    events: AuditEvent[];
  };
};
