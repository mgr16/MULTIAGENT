import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from app.eval.harness import run_dataset, run_ab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Ruta a prompts.jsonl")
    ap.add_argument("--runs", type=int, default=1, help="Repeticiones por prompt (self-consistency simple)")
    ap.add_argument("--ab", action="store_true", help="Corre A/B")
    ap.add_argument("--a", type=str, default=None, help="Overrides JSON para A")
    ap.add_argument("--b", type=str, default=None, help="Overrides JSON para B")
    args = ap.parse_args()

    if args.ab:
        if not args.a or not args.b:
            raise SystemExit("--ab requiere --a y --b con JSON de overrides")
        overrides_a = json.loads(args.a)
        overrides_b = json.loads(args.b)
        out = asyncio.run(run_ab(Path(args.dataset), overrides_a, overrides_b, runs=args.runs))
    else:
        out = asyncio.run(run_dataset(Path(args.dataset), runs=args.runs))

    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
