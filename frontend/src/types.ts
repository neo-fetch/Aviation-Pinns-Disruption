// Shared message/data types mirroring server/app.py payloads.

export interface NetworkNode {
  id: number;
  name: string;
  tier: string;
  tier_idx: number;
  region: string;
  prod_capacity: number;
  storage_capacity: number;
  reliability: number;
  is_sole_source: boolean;
  in_degree: number;
  out_degree: number;
}

export interface NetworkEdge {
  source: number;
  target: number;
  lead_time: number;
  transport_capacity: number;
}

export interface NetworkMeta {
  horizons: number[];
  window_weeks: number;
  severity_thresholds: number[];
  severity_classes: string[];
  disruption_types: string[];
  tiers: string[];
}

export interface VulnerabilityEntry {
  node: number;
  name: string;
  tier: string;
  score: number;
  betweenness: number;
  sole_source: boolean;
}

export interface NetworkResponse {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  measures: Record<string, number>;
  vulnerability_ranking: VulnerabilityEntry[];
  meta: NetworkMeta;
}

export interface EpisodeInfo {
  id?: number;
  kind: string;
  node: number;
  start: number;
  duration: number;
  magnitude: number;
  pre_weeks: number;
  recovery_weeks: number;
  intensity?: number;
}

export interface RiskBlock {
  horizons: number[];
  risk: number[][]; // [horizon][node] P(moderate or major)
  severity: number[][]; // [horizon][node] argmax class
}

export interface WeekFrame {
  type: "week";
  week: number;
  nodes: {
    inventory_frac: number[];
    utilization: number[];
    days_supply: number[];
    backlog: number[];
    arrivals: number[];
    cap_reduction: number[];
    severity: number[];
  };
  edges: { flow_frac: number[] };
  active_episodes: EpisodeInfo[];
  risk: RiskBlock | null;
  model_state: string;
}

export interface ModelStatus {
  state: "untrained" | "training" | "ready" | "failed";
  quality: string | null;
  val_f1: number | null;
  horizons: number[];
  window_weeks: number;
  error: string | null;
  training_seconds?: number;
}

export interface HelloMsg {
  type: "hello";
  seed: number;
  week: number;
  playing: boolean;
  speed: number;
  auto_episodes: boolean;
  episodes: EpisodeInfo[];
  model: ModelStatus;
}

export interface StatusMsg {
  type: "status";
  playing: boolean;
  speed: number;
  week: number;
  seed: number;
}

export interface InjectedMsg {
  type: "injected";
  episode: EpisodeInfo;
}

export interface ErrorMsg {
  type: "error";
  message: string;
}

export type ServerMsg = HelloMsg | WeekFrame | StatusMsg | InjectedMsg | ErrorMsg;

export interface InjectRequest {
  kind: string;
  node: number;
  magnitude: number;
  duration: number;
  start_offset: number;
}
