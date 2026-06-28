"""
End-to-end test of the chat layer against a battery of real questions.

Exercises both the structured path (track position lookups) and the
semantic path (fuzzy questions). Prints PASS/FAIL summary.

Usage:
    python e2e_test.py
    python e2e_test.py --semantic   # also test the semantic path
                            (requires embeddings to be built)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from chat import answer_question


# Test cases: (question, expected substring in answer, kind)
# 'kind' is "structured" (should hit lookup_track) or "semantic" (semantic_search)
TESTS = [
    # === KILLER TEST ===
    (
        "What was the second track of Minimoni's second album?",
        "CRAZY ABOUT YOU",
        "structured",
    ),
    # === Variations of the structured query ===
    (
        "What was the 2nd track of Minimoni's 2nd album?",
        "CRAZY ABOUT YOU",
        "structured",
    ),
    (
        "2nd track on Morning Musume 5th album",
        "Summer Night Town",
        "structured",
    ),
    (
        "What was the third track of Berryz Koubou's first album?",
        "Nicchoku",
        "structured",
    ),
    # === Listing paths ===
    (
        "List all albums by Morning Musume",
        "First Time",
        "list",
    ),
    (
        "List all albums by Berryz Koubou",
        "Berryz",
        "list",
    ),
    # === Error handling ===
    (
        "What was the 99th track of Minimoni's 2nd album?",
        "tracks",
        "structured",  # should return an error message
    ),
    # === Semantic queries (need embeddings) ===
    (
        "Who produced CRAZY ABOUT YOU?",
        "Tsunku",
        "semantic",
    ),
    (
        "Tell me about Minimoni",
        "Minimoni",
        "semantic",
    ),
]


def run(db_path: Path, chroma_dir: Path, include_semantic: bool = True, use_llm: bool = False) -> int:
    # Set up LLM synthesizer. When use_llm=False we explicitly disable it
    # so test results stay deterministic and don't depend on API keys.
    llm = None
    if use_llm:
        try:
            from llm import LLMSynthesizer
            llm = LLMSynthesizer()
        except Exception:
            llm = None

    failures = 0
    runs = 0
    print(f"\n{'='*70}")
    llm_status = llm.describe() if llm else "LLM: disabled"
    print(f"Running {len(TESTS)} tests (semantic={include_semantic}, {llm_status})")
    print(f"{'='*70}\n")

    for i, (q, expected, kind) in enumerate(TESTS, start=1):
        if kind == "semantic" and not include_semantic:
            continue
        runs += 1
        try:
            answer = answer_question(q, db_path, chroma_dir, llm=llm)
        except Exception as e:
            print(f"  [{i}] FAIL  q={q!r}")
            print(f"        exception: {e}")
            failures += 1
            continue

        # Determine pass/fail
        # For error tests, we just check that *some* response was produced
        # containing 'tracks' or 'error' indicator.
        ok = expected.lower() in answer.lower()
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1

        print(f"  [{i}] {status}  ({kind})  q={q!r}")
        if not ok:
            print(f"        expected substring: {expected!r}")
            print(f"        answer:")
            for line in answer.splitlines()[:8]:
                print(f"          {line}")
        print()

    print(f"\n{'='*70}")
    print(f"Summary: {runs - failures}/{runs} passed, {failures} failed")
    print(f"{'='*70}")
    return 0 if failures == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(HERE / "helloproject.db"))
    p.add_argument("--chroma", default=str(HERE / "chroma"))
    p.add_argument("--structured-only", action="store_true",
                   help="Skip semantic tests (don't need embeddings)")
    p.add_argument("--use-llm", action="store_true",
                   help="Enable the LLM synthesis layer (auto-detects provider)")
    p.add_argument("--llm-status", action="store_true",
                   help="Just print the LLM provider status and exit")
    args = p.parse_args()

    db_path = Path(args.db)
    chroma_dir = Path(args.chroma)

    if not db_path.exists():
        print(f"No DB at {db_path}. Run build_index.py first.")
        return 1

    if args.llm_status:
        try:
            from llm import LLMSynthesizer
            s = LLMSynthesizer()
            print(s.describe())
            print(f"  available: {s.available}")
        except Exception as e:
            print(f"LLM init failed: {e}")
            return 1
        return 0

    return run(db_path, chroma_dir, include_semantic=not args.structured_only, use_llm=args.use_llm)


if __name__ == "__main__":
    sys.exit(main())