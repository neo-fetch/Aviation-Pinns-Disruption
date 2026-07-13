"""Persistence and validation for user-defined ("custom") network nodes.

Custom node definitions live in data/custom_nodes.json and are appended to
the built-in Airbus network on every (re)build — see
pignn.network.apply_custom_nodes. The list order in the file is
authoritative: ids are assigned in that order after the built-in nodes.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

import networkx as nx

from pignn.network import TIER_ORDER, TIERS

log = logging.getLogger("custom_nodes")

REGIONS = ("EU", "NA", "APAC", "CIS", "AFR", "MEA", "GLOBAL")
MAX_NAME_LEN = 60
MAX_PROD = 2000.0
MAX_STOR = 10000.0

OVERRIDE_FIELDS = ("upstream", "downstream",
                   "exclude_upstream", "exclude_downstream")


class CustomNodeStore:
    """JSON-file-backed list of custom node defs with atomic writes."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> list:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text())
            nodes = payload.get("nodes", [])
            if not isinstance(nodes, list):
                raise ValueError("'nodes' is not a list")
            return nodes
        except (ValueError, OSError) as e:
            log.warning("ignoring corrupt custom node store %s: %s",
                        self.path, e)
            return []

    def save(self, defs: list) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"version": 1, "nodes": defs}, f, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def add(self, d: dict) -> None:
        defs = self.load()
        defs.append(d)
        self.save(defs)

    def remove(self, name: str) -> dict:
        defs = self.load()
        for i, d in enumerate(defs):
            if d["name"] == name:
                removed = defs.pop(i)
                self.save(defs)
                return removed
        raise KeyError(name)

    def referenced_by(self, name: str) -> list:
        """Names of other custom defs whose overrides mention this node."""
        return [d["name"] for d in self.load()
                if d["name"] != name
                and any(name in d.get(f, []) for f in OVERRIDE_FIELDS)]


def normalize_def(d: dict) -> dict:
    """Keep only known fields, with defaults filled in."""
    out = {
        "name": str(d.get("name", "")).strip(),
        "tier": d.get("tier"),
        "region": d.get("region"),
        "prod_capacity": d.get("prod_capacity"),
        "storage_capacity": d.get("storage_capacity"),
        "reliability": d.get("reliability"),
        "is_sole_source": bool(d.get("is_sole_source", False)),
    }
    for f in OVERRIDE_FIELDS:
        vals = d.get(f) or []
        out[f] = [str(v) for v in vals if str(v).strip()]
    return out


def validate_def(d: dict, G: nx.DiGraph) -> list:
    """Return a list of error strings; empty means the def is acceptable.

    Validates against the current graph (built-ins + already-applied
    customs) so names stay unique and overrides reference real nodes.
    """
    errors = []
    name = d["name"]
    if not name:
        errors.append("name must not be empty")
    elif len(name) > MAX_NAME_LEN:
        errors.append(f"name must be at most {MAX_NAME_LEN} characters")
    elif "/" in name:
        errors.append("name must not contain '/'")

    existing = {G.nodes[n]["name"]: n for n in G.nodes}
    if name and name.lower() in {k.lower() for k in existing}:
        errors.append(f"a node named '{name}' already exists")

    tier = d["tier"]
    if tier not in TIERS:
        errors.append(f"tier must be one of {list(TIERS)}")
        return errors  # tier-dependent checks below would be meaningless
    if d["region"] not in REGIONS:
        errors.append(f"region must be one of {list(REGIONS)}")

    if tier != "customer":
        for field, hi, label in (("prod_capacity", MAX_PROD, "production"),
                                 ("storage_capacity", MAX_STOR, "storage")):
            v = d[field]
            if not isinstance(v, (int, float)) or not 0 < float(v) <= hi:
                errors.append(f"{label} capacity must be in (0, {hi:g}]")
        r = d["reliability"]
        if not isinstance(r, (int, float)) or not 0.8 <= float(r) <= 1.0:
            errors.append("reliability must be in [0.8, 1.0]")

    ti = TIER_ORDER.index(tier)
    for f in OVERRIDE_FIELDS:
        upstream = f.endswith("upstream")
        for ref in d[f]:
            n = existing.get(ref)
            if n is None:
                errors.append(f"{f}: unknown node '{ref}'")
                continue
            rt = G.nodes[n]["tier"]
            rti = TIER_ORDER.index(rt)
            if upstream and (rti >= ti or rt == "customer"):
                errors.append(f"{f}: '{ref}' ({rt}) must be in a lower tier "
                              f"than {tier} and not a customer")
            if not upstream and (rti <= ti or rt == "raw"):
                errors.append(f"{f}: '{ref}' ({rt}) must be in a higher tier "
                              f"than {tier} and not raw")
    return errors
