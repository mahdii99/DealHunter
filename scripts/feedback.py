"""Messages from the complaints & suggestions box on the page, newest first.

    python -m scripts.feedback          # last 50
    python -m scripts.feedback 200      # last 200
"""
import sys
from collections import Counter
from datetime import datetime

from app import db


def main(n: int) -> None:
    rows = db.list_feedback(n)
    if not rows:
        print("no feedback yet")
        return
    print("by kind:", dict(Counter(r["kind"] for r in rows)))
    for r in rows:
        print(f"\n{datetime.fromtimestamp(r['at']):%Y-%m-%d %H:%M}  {r['kind']:<10} {r['contact'] or '-'}")
        print(f"    {r['message']}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
