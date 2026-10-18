"""Evaluate structured suggestions using the running local FastAPI backend.

Usage (from repository root):
  services\\writing-engine\\.venv\\Scripts\\python.exe tests\\run_spelling_benchmark.py
"""
import json
import pathlib
import time
import urllib.error
import urllib.request

CASES = pathlib.Path(__file__).with_name("spelling_cases.json")
URL = "http://127.0.0.1:8765/v1/analyze"


def run() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    total_expected = total_matched = total_returned = 0
    for case in cases:
        body = json.dumps({"text": case["text"]}).encode("utf-8")
        req = urllib.request.Request(
            URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.load(resp)
        except (urllib.error.URLError, TimeoutError) as error:
            print(f'{case["id"]}: ERROR {error}')
            continue
        elapsed = time.perf_counter() - start
        actual = {(s["original"], s["replacement"]) for s in data["suggestions"]}
        expected = {(s["original"], s["replacement"]) for s in case["expected"]}
        matched = actual & expected
        total_expected += len(expected)
        total_matched += len(matched)
        total_returned += len(actual)
        print(f'{case["id"]}: {len(matched)}/{len(expected)} expected, '
              f'{len(actual)} returned, {elapsed:.2f}s')
        for missed in sorted(expected - actual):
            print(f'  MISSED: {missed[0]} -> {missed[1]}')
        for extra in sorted(actual - expected):
            print(f'  REVIEW: {extra[0]} -> {extra[1]}')
    recall = total_matched / total_expected if total_expected else 1
    precision = total_matched / total_returned if total_returned else 1
    print(f'\nSample recall: {recall:.1%}; exact-match precision: {precision:.1%}')
    print("Small diagnostic dataset only; inspect suggested explanations manually.")


if __name__ == "__main__":
    run()
