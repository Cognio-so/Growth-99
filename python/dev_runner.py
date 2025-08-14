# dev_runner.py — tiny CLI to call main_app.handle_request
from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path

# Ensure we run from project root
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import main_app  # noqa: E402

async def _run(method: str, path: str, body: dict | None):
    res = await main_app.handle_request(method, path, body or {})
    # If an SSE-like stream (async iterator), print chunks as they arrive
    if hasattr(res, "__aiter__"):
        async for chunk in res:
            if isinstance(chunk, (bytes, bytearray)):
                sys.stdout.buffer.write(chunk)
            else:
                sys.stdout.write(str(chunk))
            sys.stdout.flush()
        return
    # Otherwise it's a plain dict
    print(json.dumps(res, indent=2, ensure_ascii=False))

def main():
    p = argparse.ArgumentParser(description="Call main_app routes without a web server")
    p.add_argument("method", choices=["GET", "POST", "DELETE"], help="HTTP verb")
    p.add_argument("path", help="Route path, e.g. /api/build-from-url")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--json", help="Inline JSON body string")
    g.add_argument("--json-file", help="Path to a JSON file for body")
    args = p.parse_args()

    body = None
    if args.json:
        body = json.loads(args.json)
    elif args.json_file:
        body = json.loads(Path(args.json_file).read_text(encoding="utf-8"))

    asyncio.run(_run(args.method, args.path, body))

if __name__ == "__main__":
    main()
