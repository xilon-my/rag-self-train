"""CLI demo: python scripts/run_demo_cli.py "问题"  (optionally --render to save page images)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from demo.engine import DemoEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--render", action="store_true", help="render page images to /tmp")
    args = ap.parse_args()

    print("loading engine (corpus + trained models) ...")
    eng = DemoEngine()
    hits = eng.retrieve(args.query)
    print(f"\nquery: {args.query}\n")
    for h in hits:
        print(f"  #{hits.index(h)+1} [{h['doc']} p{h['page']}] score={h['score']:.3f}")
        print(f"     {h['text'][:100]}...")
        if args.render:
            eng.render_page(h["doc"], h["page"], f"/tmp/demo_p{h['page']}.png")
    print("\n(top-5 pages above; run with --render to save page images)")


if __name__ == "__main__":
    main()
