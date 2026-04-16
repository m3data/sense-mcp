#!/usr/bin/env python3
"""Sense retrieval eval runner.

Runs ground-truth queries against the live Sense index and reports
precision@k, recall of expected files, and ranking violations.

Usage:
    python evals/eval_runner.py                    # run all, print report
    python evals/eval_runner.py --json             # machine-readable output
    python evals/eval_runner.py --tag baseline     # tag results for comparison
    python evals/eval_runner.py --compare baseline # compare current vs tagged run

Requires: sense-mcp installed (editable or otherwise) and sense.db populated.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add parent to path for sense_mcp imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL_QUERIES_PATH = Path(__file__).parent / "retrieval_eval_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_queries(path: Path | None = None) -> list[dict]:
    path = path or EVAL_QUERIES_PATH
    with open(path) as f:
        data = json.load(f)
    return data["queries"]


def run_query(query: dict, ecosystem_root: str) -> dict:
    """Run a single eval query against the live Sense index."""
    from sense_mcp import server as sense_server
    from sense_mcp.session import SessionState

    query_text = query["query"]
    mode = query.get("mode")
    k = query.get("k", 5)

    # Embed query
    query_emb = sense_server.embed_query(query_text)

    # Run mode-aware search
    results, meta = sense_server.search_chunks_contextual(
        query_emb,
        query_text=query_text,
        mode=mode if mode else "none",
        limit=k,
    )

    # Normalise file paths to relative
    result_files = []
    for r in results:
        rel = r["file_path"].replace(ecosystem_root + "/", "")
        result_files.append({
            "file_path": rel,
            "section": r.get("section", ""),
            "score": round(r.get("score", 0.0), 4),
            "similarity": round(r.get("similarity", 0.0), 4),
            "source_type": r.get("source_type", ""),
            "project": r.get("project", ""),
            "slot_type": r.get("slot_type", ""),
            "resurfaced": r.get("resurfaced", False),
            "resurface_penalty": round(r.get("resurface_penalty", 1.0), 4),
            "mode_multiplier": round(r.get("mode_multiplier", 1.0), 4),
            "cross_project": r.get("cross_project", False),
        })

    # Evaluate against expectations
    should_surface = query.get("should_surface", [])
    should_not_outrank_files = query.get("should_not_outrank", {}).get("files", [])

    surfaced_paths = [r["file_path"] for r in result_files]

    # Recall: how many expected files appeared in top-k?
    hits = [f for f in should_surface if f in surfaced_paths]
    misses = [f for f in should_surface if f not in surfaced_paths]
    recall = len(hits) / len(should_surface) if should_surface else 1.0

    # Ranking violations: did any should_not_outrank file appear above a should_surface file?
    violations = []
    for bad_file in should_not_outrank_files:
        if bad_file in surfaced_paths:
            bad_rank = surfaced_paths.index(bad_file)
            for good_file in should_surface:
                if good_file in surfaced_paths:
                    good_rank = surfaced_paths.index(good_file)
                    if bad_rank < good_rank:
                        violations.append({
                            "bad_file": bad_file,
                            "bad_rank": bad_rank + 1,
                            "good_file": good_file,
                            "good_rank": good_rank + 1,
                        })
                else:
                    # Good file missing entirely, bad file present = violation
                    violations.append({
                        "bad_file": bad_file,
                        "bad_rank": bad_rank + 1,
                        "good_file": good_file,
                        "good_rank": "missing",
                    })

    return {
        "id": query["id"],
        "query": query_text,
        "mode": mode,
        "k": k,
        "results": result_files,
        "expected_hits": hits,
        "expected_misses": misses,
        "recall": round(recall, 3),
        "violations": violations,
        "meta": {
            "mode_used": meta.get("mode"),
            "diversity_profile": meta.get("diversity_profile"),
            "trajectory": meta.get("trajectory", {}),
        },
    }


def run_all(queries: list[dict]) -> dict:
    """Run all eval queries and return aggregate results."""
    import sqlite3
    from sense_mcp import server as sense_server
    from sense_mcp.config import get_config

    cfg = get_config()
    ecosystem_root = str(cfg.root)

    # Ensure DB is connected
    if sense_server._db_conn is None:
        db_path = str(cfg.db_path)
        if not os.path.exists(db_path):
            print(f"ERROR: Database not found at {db_path}", file=sys.stderr)
            sys.exit(1)
        sense_server._db_conn = sqlite3.connect(db_path)
        sense_server._db_conn.execute("PRAGMA journal_mode=WAL")

    # Clear session state so surfacing penalties don't accumulate across eval queries
    from sense_mcp.session import SessionState, STATE_PATH
    clean_state = SessionState()
    clean_state.save()

    results = []
    for q in queries:
        # Reset session state between queries to avoid cross-contamination
        clean_state = SessionState()
        clean_state.save()

        result = run_query(q, ecosystem_root)
        results.append(result)

    # Aggregate stats
    total_recall = sum(r["recall"] for r in results) / len(results) if results else 0
    total_violations = sum(len(r["violations"]) for r in results)
    perfect_queries = sum(1 for r in results if r["recall"] == 1.0 and not r["violations"])

    # Group by mode
    by_mode = {}
    for r in results:
        mode = r["mode"] or "flat"
        if mode not in by_mode:
            by_mode[mode] = {"recall_sum": 0, "count": 0, "violations": 0}
        by_mode[mode]["recall_sum"] += r["recall"]
        by_mode[mode]["count"] += 1
        by_mode[mode]["violations"] += len(r["violations"])

    mode_stats = {}
    for mode, stats in by_mode.items():
        mode_stats[mode] = {
            "mean_recall": round(stats["recall_sum"] / stats["count"], 3),
            "violations": stats["violations"],
            "queries": stats["count"],
        }

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_queries": len(results),
        "mean_recall": round(total_recall, 3),
        "total_violations": total_violations,
        "perfect_queries": perfect_queries,
        "by_mode": mode_stats,
        "results": results,
    }


def print_report(run: dict):
    """Print human-readable eval report."""
    print("=" * 72)
    print(f"  SENSE RETRIEVAL EVAL — {run['timestamp']}")
    print("=" * 72)
    print()
    print(f"  Queries: {run['total_queries']}  |  "
          f"Mean recall: {run['mean_recall']:.1%}  |  "
          f"Violations: {run['total_violations']}  |  "
          f"Perfect: {run['perfect_queries']}/{run['total_queries']}")
    print()

    # Per-mode summary
    print("  By mode:")
    for mode, stats in sorted(run["by_mode"].items()):
        print(f"    {mode:12s}  recall={stats['mean_recall']:.1%}  "
              f"violations={stats['violations']}  queries={stats['queries']}")
    print()

    # Per-query detail
    for r in run["results"]:
        status = "PASS" if r["recall"] == 1.0 and not r["violations"] else "FAIL"
        mode_str = r["mode"] or "flat"
        print(f"  [{status}] {r['id']} ({mode_str}): {r['query'][:50]}")
        print(f"         recall={r['recall']:.0%}  hits={len(r['expected_hits'])}  "
              f"misses={len(r['expected_misses'])}  violations={len(r['violations'])}")

        if r["expected_misses"]:
            for m in r["expected_misses"]:
                print(f"         MISS: {m}")

        if r["violations"]:
            for v in r["violations"]:
                print(f"         VIOLATION: {v['bad_file']} (rank {v['bad_rank']}) "
                      f"outranked {v['good_file']} (rank {v['good_rank']})")

        # Show top results briefly
        for i, res in enumerate(r["results"][:5], 1):
            flag = ""
            if res["file_path"] in r["expected_hits"]:
                flag = " ✓"
            elif res["file_path"] in [v["bad_file"] for v in r["violations"]]:
                flag = " ✗"
            print(f"         {i}. score={res['score']:.3f} sim={res['similarity']:.3f} "
                  f"{res['file_path'][:60]}{flag}")

        print()

    print("=" * 72)


def save_results(run: dict, tag: str | None = None):
    """Save results to disk for later comparison."""
    RESULTS_DIR.mkdir(exist_ok=True)
    if tag:
        path = RESULTS_DIR / f"eval_{tag}.json"
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"eval_{ts}.json"

    with open(path, "w") as f:
        json.dump(run, f, indent=2)
    print(f"\nResults saved to {path}")
    return path


def compare_runs(current: dict, baseline: dict):
    """Print comparison between two eval runs."""
    print()
    print("=" * 72)
    print("  COMPARISON: current vs baseline")
    print("=" * 72)
    print()
    print(f"  {'Metric':<25s} {'Baseline':>10s} {'Current':>10s} {'Delta':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    br = baseline["mean_recall"]
    cr = current["mean_recall"]
    print(f"  {'Mean recall':<25s} {br:>9.1%} {cr:>9.1%} {cr-br:>+9.1%}")

    bv = baseline["total_violations"]
    cv = current["total_violations"]
    print(f"  {'Violations':<25s} {bv:>10d} {cv:>10d} {cv-bv:>+10d}")

    bp = baseline["perfect_queries"]
    cp = current["perfect_queries"]
    print(f"  {'Perfect queries':<25s} {bp:>10d} {cp:>10d} {cp-bp:>+10d}")

    print()

    # Per-mode comparison
    all_modes = sorted(set(list(baseline["by_mode"].keys()) + list(current["by_mode"].keys())))
    print(f"  {'Mode':<12s} {'Base recall':>12s} {'Curr recall':>12s} {'Delta':>10s}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    for mode in all_modes:
        bs = baseline["by_mode"].get(mode, {})
        cs = current["by_mode"].get(mode, {})
        br = bs.get("mean_recall", 0)
        cr = cs.get("mean_recall", 0)
        print(f"  {mode:<12s} {br:>11.1%} {cr:>11.1%} {cr-br:>+9.1%}")

    print()

    # Per-query deltas
    baseline_by_id = {r["id"]: r for r in baseline["results"]}
    print("  Per-query changes:")
    for cr in current["results"]:
        br = baseline_by_id.get(cr["id"])
        if not br:
            print(f"    {cr['id']}: NEW (recall={cr['recall']:.0%})")
            continue
        if br["recall"] != cr["recall"] or len(br["violations"]) != len(cr["violations"]):
            print(f"    {cr['id']}: recall {br['recall']:.0%} → {cr['recall']:.0%}  "
                  f"violations {len(br['violations'])} → {len(cr['violations'])}")

    print()
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Sense retrieval eval runner")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of report")
    parser.add_argument("--tag", type=str, help="Tag this run for later comparison")
    parser.add_argument("--compare", type=str, help="Compare current run against tagged baseline")
    parser.add_argument("--queries", type=str, help="Path to eval queries JSON")
    args = parser.parse_args()

    queries_path = Path(args.queries) if args.queries else None
    queries = load_queries(queries_path)

    print(f"Running {len(queries)} eval queries...", file=sys.stderr)
    run = run_all(queries)

    if args.json:
        print(json.dumps(run, indent=2))
    else:
        print_report(run)

    # Save results
    save_path = save_results(run, tag=args.tag)

    # Compare if requested
    if args.compare:
        baseline_path = RESULTS_DIR / f"eval_{args.compare}.json"
        if not baseline_path.exists():
            print(f"ERROR: Baseline '{args.compare}' not found at {baseline_path}",
                  file=sys.stderr)
            sys.exit(1)
        with open(baseline_path) as f:
            baseline = json.load(f)
        compare_runs(run, baseline)


if __name__ == "__main__":
    main()
