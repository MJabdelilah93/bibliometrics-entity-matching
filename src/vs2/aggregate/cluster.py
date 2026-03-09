"""cluster.py — C7 entity aggregation via union-find over MATCH edges.

Design rules (B5)
-----------------
1. MATCH edges only are used to merge nodes into connected components.
2. UNCERTAIN edges are ignored (do not merge, do not conflict).
3. If any NON-MATCH edge exists whose two endpoints land in the SAME component,
   that component is flagged CONFLICTED and routed for human adjudication.

Entity IDs
----------
Each component gets a deterministic entity_id derived from the sha256 of its
sorted member node IDs, encoded as UTF-8 hex.  This is stable across reruns
given the same match-edge set.

Usage
-----
    from vs2.aggregate.cluster import build_clusters_from_routing, detect_conflicts

    membership_df, stats = build_clusters_from_routing(routing_df, task="AND")
    conflicts_df, summary_df = detect_conflicts(routing_df, membership_df, task="AND")
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Union-find (path-compressed, union-by-rank)
# ---------------------------------------------------------------------------

class _UF:
    """Disjoint-set structure with path compression and union by rank."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank:   dict[str, int] = {}

    def add(self, node: str) -> None:
        if node not in self._parent:
            self._parent[node] = node
            self._rank[node]   = 0

    def find(self, node: str) -> str:
        # Path compression
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


def union_find(
    nodes: Iterable[str],
    match_edges: Iterable[tuple[str, str]],
) -> dict[str, str]:
    """Run union-find and return a deterministic node → entity_id mapping.

    entity_id is the sha256 hex of the sorted, newline-joined member node IDs.
    This is stable given the same input set.

    Parameters
    ----------
    nodes       : All node IDs to include (even isolated singletons).
    match_edges : (node_a, node_b) pairs to merge — MATCH label only.

    Returns
    -------
    dict mapping each node_id -> entity_id string.
    """
    uf = _UF()
    for n in nodes:
        uf.add(n)
    # Sort edges for determinism before processing
    for a, b in sorted(match_edges):
        uf.union(a, b)

    # Group members by root
    root_to_members: dict[str, list[str]] = defaultdict(list)
    for n in uf._parent:
        root_to_members[uf.find(n)].append(n)

    # Assign deterministic entity_id per component
    node_to_entity: dict[str, str] = {}
    for members in root_to_members.values():
        members_sorted = sorted(members)
        digest = hashlib.sha256(
            "\n".join(members_sorted).encode("utf-8")
        ).hexdigest()
        for n in members_sorted:
            node_to_entity[n] = digest
    return node_to_entity


# ---------------------------------------------------------------------------
# Build clusters from routing DataFrame
# ---------------------------------------------------------------------------

def build_clusters_from_routing(
    routing_df: pd.DataFrame,
    task: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build entity clusters from a benchmark-filtered routing DataFrame.

    Parameters
    ----------
    routing_df : DataFrame with columns anchor_id, candidate_id, label_final.
                 Typically loaded from routing_log_{task}_bm.parquet.
    task       : "AND" or "AIN" (written into membership_df for traceability).

    Returns
    -------
    (membership_df, cluster_stats)

    membership_df columns
    ---------------------
        task        str   task label
        entity_id   str   sha256 hex of sorted member ids
        node_id     str   individual instance/author ID

    cluster_stats keys
    ------------------
        task, n_nodes, n_match_edges, n_clusters,
        n_singleton_clusters, largest_cluster_size,
        top20_sizes  (list of int, descending)
    """
    nodes: set[str] = set()
    nodes.update(routing_df["anchor_id"].tolist())
    nodes.update(routing_df["candidate_id"].tolist())

    match_mask = routing_df["label_final"] == "match"
    match_edges: list[tuple[str, str]] = list(
        zip(
            routing_df.loc[match_mask, "anchor_id"],
            routing_df.loc[match_mask, "candidate_id"],
        )
    )

    node_to_entity = union_find(nodes, match_edges)

    # Build membership DataFrame
    rows = [
        {"task": task, "entity_id": eid, "node_id": nid}
        for nid, eid in sorted(node_to_entity.items())
    ]
    membership_df = pd.DataFrame(rows, columns=["task", "entity_id", "node_id"])

    # Cluster size distribution
    sizes = membership_df.groupby("entity_id").size().sort_values(ascending=False)
    top20 = sizes.head(20).tolist()
    n_singletons = int((sizes == 1).sum())

    cluster_stats: dict[str, Any] = {
        "task":                task,
        "n_nodes":             len(nodes),
        "n_match_edges":       len(match_edges),
        "n_clusters":          int(sizes.shape[0]),
        "n_singleton_clusters": n_singletons,
        "largest_cluster_size": int(top20[0]) if top20 else 0,
        "top20_sizes":         top20,
    }
    return membership_df, cluster_stats


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def detect_conflicts(
    routing_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    task: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect internal non-match edges within merged clusters.

    A conflict occurs when a NON-MATCH routing decision connects two nodes
    that were merged into the same entity by MATCH edges.

    Parameters
    ----------
    routing_df    : Same routing DataFrame passed to build_clusters_from_routing.
    membership_df : Output of build_clusters_from_routing (node_id -> entity_id).
    task          : "AND" or "AIN".

    Returns
    -------
    (conflicts_df, conflicts_summary_df)

    conflicts_df columns
    --------------------
        task, entity_id, node_a, node_b, label_final

    conflicts_summary_df columns
    ----------------------------
        task, entity_id, num_members, num_internal_nonmatch_edges,
        sample_edges  (JSON string, up to 5 edges as [[node_a, node_b], ...])
    """
    node_to_entity = dict(zip(membership_df["node_id"], membership_df["entity_id"]))

    nm_mask = routing_df["label_final"] == "non-match"
    nm_df = routing_df.loc[nm_mask, ["anchor_id", "candidate_id", "label_final"]].copy()

    nm_df["entity_a"] = nm_df["anchor_id"].map(node_to_entity)
    nm_df["entity_b"] = nm_df["candidate_id"].map(node_to_entity)

    # Internal conflict: both endpoints in the same entity
    conflict_mask = nm_df["entity_a"] == nm_df["entity_b"]
    internal = nm_df[conflict_mask].copy()

    if internal.empty:
        conflicts_df = pd.DataFrame(
            columns=["task", "entity_id", "node_a", "node_b", "label_final"]
        )
        conflicts_summary_df = pd.DataFrame(
            columns=["task", "entity_id", "num_members",
                     "num_internal_nonmatch_edges", "sample_edges"]
        )
        return conflicts_df, conflicts_summary_df

    conflicts_df = pd.DataFrame({
        "task":        task,
        "entity_id":   internal["entity_a"].values,
        "node_a":      internal["anchor_id"].values,
        "node_b":      internal["candidate_id"].values,
        "label_final": "non-match",
    })

    # Summary: one row per conflicted entity
    entity_sizes = (
        membership_df.groupby("entity_id")
        .size()
        .rename("num_members")
        .reset_index()
    )
    edge_counts = (
        conflicts_df.groupby("entity_id")
        .size()
        .rename("num_internal_nonmatch_edges")
        .reset_index()
    )
    # Sample edges (up to 5) as JSON
    sample_rows = []
    for eid, grp in conflicts_df.groupby("entity_id"):
        edges = list(zip(grp["node_a"].tolist(), grp["node_b"].tolist()))
        sample_rows.append({
            "entity_id":    eid,
            "sample_edges": json.dumps(edges[:5]),
        })
    sample_df = pd.DataFrame(sample_rows)

    summary = (
        entity_sizes
        .merge(edge_counts, on="entity_id")
        .merge(sample_df,   on="entity_id")
    )
    summary.insert(0, "task", task)
    conflicts_summary_df = summary[
        ["task", "entity_id", "num_members",
         "num_internal_nonmatch_edges", "sample_edges"]
    ]

    return conflicts_df, conflicts_summary_df


# ---------------------------------------------------------------------------
# Run C7 for one task: load, cluster, detect, write outputs
# ---------------------------------------------------------------------------

def run_c7_task(
    task: str,
    bm_routing_path: str | Path,
    outputs_dir: str | Path,
    manifests_dir: str | Path,
) -> dict[str, Any]:
    """Full C7 aggregation for one task.

    Parameters
    ----------
    task             : "AND" or "AIN".
    bm_routing_path  : Path to routing_log_{task}_bm.parquet.
    outputs_dir      : Directory for CSV outputs.
    manifests_dir    : Directory for manifest JSON (shared across tasks).

    Returns
    -------
    dict with cluster_stats, n_conflicted_entities, output_paths.
    """
    t = task.lower()
    bm_path   = Path(bm_routing_path)
    out_dir   = Path(outputs_dir)
    man_dir   = Path(manifests_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    routing_df = pd.read_parquet(bm_path)

    # --- cluster ---
    membership_df, cluster_stats = build_clusters_from_routing(routing_df, task)

    # --- conflicts ---
    conflicts_df, conflicts_summary_df = detect_conflicts(
        routing_df, membership_df, task
    )
    n_conflicted = int(conflicts_summary_df.shape[0])

    # --- write CSVs ---
    clusters_path      = out_dir / f"clusters_{t}_bm.csv"
    conflicts_path     = out_dir / f"conflicts_{t}_bm.csv"
    conf_summary_path  = out_dir / f"conflicts_summary_{t}_bm.csv"

    membership_df[["entity_id", "node_id"]].to_csv(clusters_path,     index=False)
    conflicts_df.to_csv(conflicts_path,     index=False)
    conflicts_summary_df.to_csv(conf_summary_path, index=False)

    return {
        "task":               task,
        "bm_routing_path":    str(bm_path),
        "clusters_path":      str(clusters_path),
        "conflicts_path":     str(conflicts_path),
        "conf_summary_path":  str(conf_summary_path),
        "cluster_stats":      cluster_stats,
        "n_conflicted_entities": n_conflicted,
    }


# ---------------------------------------------------------------------------
# Write aggregation manifest
# ---------------------------------------------------------------------------

def write_aggregation_manifest(
    manifests_dir: str | Path,
    and_result: dict[str, Any] | None,
    ain_result: dict[str, Any] | None,
) -> Path:
    """Write runs/<run_id>/manifests/aggregation_manifest_bm.json."""
    man_dir  = Path(manifests_dir)
    man_dir.mkdir(parents=True, exist_ok=True)
    out_path = man_dir / "aggregation_manifest_bm.json"

    def _sha256(path: str) -> str:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _entry(result: dict[str, Any] | None) -> dict[str, Any] | None:
        if result is None:
            return None
        cs = result["cluster_stats"]
        return {
            "bm_routing_path":       result["bm_routing_path"],
            "bm_routing_sha256":     _sha256(result["bm_routing_path"]),
            "clusters_path":         result["clusters_path"],
            "conflicts_path":        result["conflicts_path"],
            "conf_summary_path":     result["conf_summary_path"],
            "n_nodes":               cs["n_nodes"],
            "n_match_edges":         cs["n_match_edges"],
            "n_clusters":            cs["n_clusters"],
            "n_singleton_clusters":  cs["n_singleton_clusters"],
            "largest_cluster_size":  cs["largest_cluster_size"],
            "top20_sizes":           cs["top20_sizes"],
            "n_conflicted_entities": result["n_conflicted_entities"],
        }

    payload: dict[str, Any] = {
        "timestamp_iso": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "and":           _entry(and_result),
        "ain":           _entry(ain_result),
    }
    out_path.write_text(
        __import__("json").dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path
