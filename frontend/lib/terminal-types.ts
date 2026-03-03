export type DataSource = "artifacts" | "demo" | "telemetry";

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
  reasoning?: ReasoningTree;
};

export type AuditTrade = {
  tradeId: string;
  asset: string;
  leg: string;
  tier: string;
  pnl: number;
  r: number;
  reason: string;
  entryTs: string;
  exitTs?: string;
};

export type AuditEvent = {
  timestamp: string;
  type: string;
  detail: string;
};

export type TerminalSnapshot = {
  meta: {
    source: DataSource;
    lastUpdated: string;
    repoRoot: string;
    modelVersion: string;
    transport: string;
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
  audit: {
    summary: string;
    trades: AuditTrade[];
    events: AuditEvent[];
  };
};
