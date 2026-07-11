"""FastAPI backend exposing the PI-GNN engine to the web frontend.

REST:
  GET  /api/network       graph structure + vulnerability ranking + meta
  GET  /api/metrics       committed batch-run metrics (outputs/metrics.json)
  GET  /api/model         model status (untrained/training/ready/failed)
  POST /api/model/train   kick off background (re)training {quality: fast|full}

WebSocket /ws/simulation — one live simulation per connection:
  client -> server: {"type": "play"} | {"type": "pause"} | {"type": "step"}
                    {"type": "set_speed", "wps": 4}
                    {"type": "reset", "seed": 7, "auto_episodes": true,
                     "n_episodes": 25}
                    {"type": "inject", "kind": "supplier_outage", "node": 40,
                     "magnitude": 0.8, "duration": 6, "start_offset": 0}
  server -> client: {"type": "hello", ...}   graph meta + episode schedule
                    {"type": "week", ...}    one simulated week (see _frame)
                    {"type": "status", ...}  playing/speed/week
                    {"type": "injected", "episode": {...}}

Run:  uvicorn server.app:app --port 8000   (from the repo root)
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as cfg
from pignn.network import build_airbus_network, node_static_features
from pignn.simulate import (DISRUPTION_TYPES, LiveSimulator, _draw_episodes)
from pignn.topology import (analytical_measures, structural_node_features,
                            vulnerability_ranking)
from server.model_service import SEVERITY_THRESHOLDS, ModelService

FRONTEND_DIST = ROOT / "frontend" / "dist"
CKPT_PATH = ROOT / cfg.OUTPUT_DIR / "pignn_live.pt"

state: dict = {}


def _episode_dict(ep, idx=None):
    d = {"kind": ep.kind, "node": ep.node, "start": ep.start,
         "duration": ep.duration, "magnitude": round(ep.magnitude, 3),
         "pre_weeks": ep.pre_weeks, "recovery_weeks": ep.recovery_weeks}
    if idx is not None:
        d["id"] = idx
    return d


@asynccontextmanager
async def lifespan(app: FastAPI):
    G = build_airbus_network(cfg.SEED)
    static = np.concatenate([node_static_features(G),
                             structural_node_features(G)], axis=1)
    state["G"] = G
    state["node_names"] = [G.nodes[i]["name"] for i in range(G.number_of_nodes())]
    state["measures"] = analytical_measures(G)
    state["ranking"] = vulnerability_ranking(G)
    svc = ModelService(G, static, CKPT_PATH)
    state["model"] = svc
    # a checkpoint from a previous run gets us risk scores immediately;
    # otherwise train a fast model in the background so the UI lights up
    # within ~a minute of first launch
    if not svc.load_checkpoint():
        svc.start_training("fast")
    yield


app = FastAPI(title="Aviation PI-GNN live backend", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------- REST
@app.get("/api/network")
def get_network():
    G = state["G"]
    nodes = []
    for i in range(G.number_of_nodes()):
        d = G.nodes[i]
        nodes.append({
            "id": i, "name": d["name"], "tier": d["tier"],
            "tier_idx": d["tier_idx"], "region": d["region"],
            "prod_capacity": round(d["prod_capacity"], 1),
            "storage_capacity": round(min(d["storage_capacity"], 1e6), 1),
            "reliability": round(d["reliability"], 3),
            "is_sole_source": d["is_sole_source"],
            "in_degree": G.in_degree(i), "out_degree": G.out_degree(i),
        })
    edges = [{"source": u, "target": v,
              "lead_time": G.edges[u, v]["lead_time"],
              "transport_capacity": round(G.edges[u, v]["transport_capacity"], 1)}
             for u, v in G.edges()]
    return {
        "nodes": nodes,
        "edges": edges,
        "measures": state["measures"],
        "vulnerability_ranking": [
            {"node": r[0], "name": r[1], "tier": r[2], "score": r[3],
             "betweenness": r[4], "sole_source": r[5]}
            for r in state["ranking"]],
        "meta": {
            "horizons": list(cfg.HORIZONS),
            "window_weeks": cfg.T_IN,
            "severity_thresholds": list(SEVERITY_THRESHOLDS),
            "severity_classes": ["none", "minor", "moderate", "major"],
            "disruption_types": list(DISRUPTION_TYPES),
            "tiers": ["raw", "tier2", "tier1", "plant", "fal", "customer"],
        },
    }


@app.get("/api/metrics")
def get_metrics():
    path = ROOT / cfg.OUTPUT_DIR / "metrics.json"
    if not path.exists():
        return JSONResponse({"error": "no batch run committed; "
                             "run scripts/run_poc.py"}, status_code=404)
    return json.loads(path.read_text())


@app.get("/api/model")
def get_model_status():
    return state["model"].status()


@app.post("/api/model/train")
def train_model_endpoint(body: dict | None = None):
    quality = (body or {}).get("quality", "fast")
    if quality not in ("fast", "full"):
        return JSONResponse({"error": "quality must be fast|full"},
                            status_code=422)
    started = state["model"].start_training(quality)
    return {"started": started, **state["model"].status()}


# -------------------------------------------------------------- WebSocket
class SimSession:
    """One live simulation per websocket connection."""

    def __init__(self, seed=None, auto_episodes=False, n_episodes=25):
        self.reset(seed, auto_episodes, n_episodes)
        self.playing = False
        self.speed = 2.0            # weeks per second

    def reset(self, seed=None, auto_episodes=False, n_episodes=25):
        G = state["G"]
        self.seed = int(seed) if seed is not None else int(
            np.random.default_rng().integers(0, 2**31))
        rng = np.random.default_rng(self.seed)
        episodes = (_draw_episodes(G, cfg.N_WEEKS, int(n_episodes), rng)
                    if auto_episodes else [])
        self.sim = LiveSimulator(G, cfg.BASE_STOCK_WEEKS,
                                 SEVERITY_THRESHOLDS, rng=rng,
                                 episodes=episodes)
        self.auto_episodes = bool(auto_episodes)

    def _frame(self, snap) -> dict:
        sim = self.sim
        dyn = snap["node_dyn"]
        risk_block = None
        svc: ModelService = state["model"]
        if sim.t >= cfg.T_IN:
            window = np.stack(sim.hist_node_dyn[-cfg.T_IN:])
            pred = svc.predict_risk(window)
            if pred is not None:
                risk, severity = pred
                risk_block = {
                    "horizons": list(cfg.HORIZONS),
                    "risk": np.round(risk, 4).tolist(),        # [H][N]
                    "severity": severity.tolist(),             # [H][N]
                }
        return {
            "type": "week",
            "week": snap["week"],
            "nodes": {
                "inventory_frac": np.round(dyn[:, 0], 4).tolist(),
                "utilization": np.round(dyn[:, 1], 4).tolist(),
                "days_supply": np.round(dyn[:, 2] * 10.0, 3).tolist(),
                "backlog": np.round(dyn[:, 3], 4).tolist(),
                "arrivals": np.round(dyn[:, 4], 4).tolist(),
                "cap_reduction": np.round(dyn[:, 5], 4).tolist(),
                "severity": snap["labels"].tolist(),
            },
            "edges": {
                "flow_frac": np.round(
                    snap["edge_flow"] / np.maximum(sim.tcap, 1e-9), 4).tolist(),
            },
            "active_episodes": [
                {**_episode_dict(ep, idx), "intensity": round(inten, 3)}
                for idx, ep, inten in snap["active_episodes"]],
            "risk": risk_block,
            "model_state": svc.state,
        }

    def step_frame(self) -> dict:
        return self._frame(self.sim.step())

    def hello(self) -> dict:
        return {
            "type": "hello",
            "seed": self.seed,
            "week": self.sim.t,
            "playing": self.playing,
            "speed": self.speed,
            "auto_episodes": self.auto_episodes,
            "episodes": [_episode_dict(ep, i)
                         for i, ep in enumerate(self.sim.episodes)],
            "model": state["model"].status(),
        }

    def status(self) -> dict:
        return {"type": "status", "playing": self.playing,
                "speed": self.speed, "week": self.sim.t,
                "seed": self.seed}


@app.websocket("/ws/simulation")
async def ws_simulation(ws: WebSocket):
    await ws.accept()
    q = ws.query_params
    session = SimSession(
        seed=q.get("seed"),
        auto_episodes=q.get("auto_episodes", "true").lower() != "false",
        n_episodes=int(q.get("n_episodes", "25")))
    send_lock = asyncio.Lock()

    async def send(payload):
        async with send_lock:
            await ws.send_json(payload)

    await send(session.hello())

    async def player():
        while True:
            if session.playing:
                frame = await asyncio.to_thread(session.step_frame)
                await send(frame)
                await asyncio.sleep(max(1.0 / session.speed, 0.02))
            else:
                await asyncio.sleep(0.05)

    task = asyncio.create_task(player())
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "play":
                session.playing = True
                await send(session.status())
            elif kind == "pause":
                session.playing = False
                await send(session.status())
            elif kind == "step":
                frame = await asyncio.to_thread(session.step_frame)
                await send(frame)
            elif kind == "set_speed":
                session.speed = float(np.clip(msg.get("wps", 2.0), 0.25, 20.0))
                await send(session.status())
            elif kind == "reset":
                session.playing = False
                session.reset(msg.get("seed"),
                              msg.get("auto_episodes", session.auto_episodes),
                              msg.get("n_episodes", 25))
                await send(session.hello())
            elif kind == "inject":
                try:
                    ep = session.sim.inject(
                        msg["kind"], int(msg["node"]),
                        float(msg.get("magnitude", 0.6)),
                        int(msg.get("duration", 6)),
                        int(msg.get("start_offset", 0)))
                    await send({"type": "injected",
                                "episode": _episode_dict(
                                    ep, len(session.sim.episodes) - 1)})
                except (KeyError, ValueError) as e:
                    await send({"type": "error", "message": str(e)})
            else:
                await send({"type": "error",
                            "message": f"unknown message type: {kind}"})
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()


# ------------------------------------------------- static frontend (built)
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"),
              name="assets")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIST / "index.html")
