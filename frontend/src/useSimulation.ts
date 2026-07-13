import { useCallback, useEffect, useRef, useState } from "react";
import type {
  EpisodeInfo,
  InjectRequest,
  ModelStatus,
  ServerMsg,
  WeekFrame,
} from "./types";

const MAX_FRAMES = 600; // ~11.5 simulated years of history for sparklines

export interface SimState {
  connected: boolean;
  playing: boolean;
  speed: number;
  nNodes: number | null;
  seed: number | null;
  week: number;
  frames: WeekFrame[];
  latest: WeekFrame | null;
  episodes: EpisodeInfo[];
  model: ModelStatus | null;
  lastError: string | null;
}

const initial: SimState = {
  connected: false,
  playing: false,
  speed: 2,
  nNodes: null,
  seed: null,
  week: 0,
  frames: [],
  latest: null,
  episodes: [],
  model: null,
  lastError: null,
};

export interface SimControls {
  play(): void;
  pause(): void;
  step(): void;
  setSpeed(wps: number): void;
  reset(opts?: { seed?: number; auto_episodes?: boolean; n_episodes?: number }): void;
  inject(req: InjectRequest): void;
}

/** WebSocket client for /ws/simulation with auto-reconnect. */
export function useSimulation(): [SimState, SimControls] {
  const [state, setState] = useState<SimState>(initial);
  const wsRef = useRef<WebSocket | null>(null);
  const closedRef = useRef(false);

  useEffect(() => {
    closedRef.current = false;
    let retry = 0;

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws/simulation`);
      wsRef.current = ws;

      ws.onopen = () => {
        retry = 0;
        setState((s) => ({ ...s, connected: true, lastError: null }));
      };
      ws.onclose = () => {
        setState((s) => ({ ...s, connected: false, playing: false }));
        if (!closedRef.current) {
          setTimeout(connect, Math.min(1000 * 2 ** retry++, 10000));
        }
      };
      ws.onmessage = (ev) => {
        const msg: ServerMsg = JSON.parse(ev.data);
        setState((s) => {
          switch (msg.type) {
            case "hello":
              return {
                ...s,
                nNodes: msg.n_nodes,
                seed: msg.seed,
                week: msg.week,
                playing: msg.playing,
                speed: msg.speed,
                episodes: msg.episodes,
                model: msg.model,
                frames: [],
                latest: null,
              };
            case "week": {
              const frames =
                s.frames.length >= MAX_FRAMES
                  ? [...s.frames.slice(-MAX_FRAMES + 1), msg]
                  : [...s.frames, msg];
              return { ...s, frames, latest: msg, week: msg.week + 1 };
            }
            case "status":
              return { ...s, playing: msg.playing, speed: msg.speed, seed: msg.seed };
            case "injected":
              return { ...s, episodes: [...s.episodes, msg.episode] };
            case "error":
              return { ...s, lastError: msg.message };
          }
        });
      };
    };

    connect();
    return () => {
      closedRef.current = true;
      wsRef.current?.close();
    };
  }, []);

  const send = useCallback((payload: object) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
  }, []);

  const controls: SimControls = {
    play: () => send({ type: "play" }),
    pause: () => send({ type: "pause" }),
    step: () => send({ type: "step" }),
    setSpeed: (wps) => send({ type: "set_speed", wps }),
    reset: (opts) => send({ type: "reset", ...opts }),
    inject: (req) => send({ type: "inject", ...req }),
  };

  return [state, controls];
}
