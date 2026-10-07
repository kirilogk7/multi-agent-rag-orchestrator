"""Command-line entry point: ``python -m rag_system "your question"``.

Exists so the system can be evaluated without opening a notebook. ``--explain``
prints the plan, the inter-agent transcript and the citation audit, which is the
fastest way to see that the coordination is real rather than decorative.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from .orchestrator import build_default_orchestrator
from .utils import DOMAINS

DEMO_QUERIES: Sequence[str] = (
    "What's the process for deploying a new microservice and what compliance "
    "checks are needed?",
    "How do I troubleshoot API performance issues while following our security "
    "policies?",
    "What business approvals are required for implementing a new data processing "
    "workflow?",
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m rag_system",
        description="Query the multi-agent RAG system over the synthetic corpus.",
    )
    parser.add_argument("query", nargs="*", help="the question to answer")
    parser.add_argument("--demo", action="store_true",
                        help="run the three assignment test scenarios")
    parser.add_argument("--explain", action="store_true",
                        help="also print the plan, agent transcript and citation audit")
    parser.add_argument("--claude", action="store_true",
                        help="use Claude for prose synthesis when ANTHROPIC_API_KEY is set")
    parser.add_argument("--expect", nargs="*", default=[], metavar="DOMAIN",
                        choices=list(DOMAINS), help="ground-truth domains, for metrics")
    parser.add_argument("--stats", action="store_true",
                        help="print the performance report when finished")
    return parser


def run(argv: Optional[List[str]] = None) -> int:
    """Answer the requested queries and print the results.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, ``2`` on an invalid query.
    """
    args = build_parser().parse_args(argv)
    queries = [" ".join(args.query)] if args.query else []
    if args.demo or not queries:
        queries = list(DEMO_QUERIES)

    orchestrator = build_default_orchestrator(use_claude=args.claude)

    for index, query in enumerate(queries):
        if index:
            print("\n" + "=" * 78 + "\n")
        try:
            if args.explain:
                plan = orchestrator.plan(query)
                print(plan.describe())
                print()
                answer = orchestrator.answer(query, expected_domains=args.expect,
                                             plan=plan)
            else:
                answer = orchestrator.answer(query, expected_domains=args.expect)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        print(answer.text)
        if args.explain:
            print()
            print(orchestrator.explain(answer.trace_id))
            print()
            audit = orchestrator.verify(answer)
            failed = [r for r in audit if not r["ok"]]
            print(f"Citation audit: {len(audit) - len(failed)}/{len(audit)} verified "
                  f"against source spans")
            for record in failed:
                print(f"  FAILED {record['marker']} {record['doc_id']}: {record['reason']}")
            # Which synthesiser actually ran. Worth printing because the
            # generative path falls back to the extractive one on any failure,
            # deliberately and silently -- so without this line there is no way
            # to tell from the output whether --claude did anything.
            print(f"Synthesizer: {answer.synthesizer}")
            print(f"Timings (ms): {json.dumps(answer.timings_ms)}")

    if args.stats:
        print("\n" + "=" * 78 + "\n")
        print(orchestrator.metrics.report())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(run())
