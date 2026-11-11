"""Results shoppers marked as wrong ("النتيجة دي غلط؟" on the page), newest first.

    python -m scripts.reports          # last 50
    python -m scripts.reports 200      # last 200
"""
import sys
from collections import Counter
from datetime import datetime

from app import db


def main(n: int) -> None:
    rows = db.list_reports(n)
    if not rows:
        print("no reports yet")
        return
    print("by reason:", dict(Counter(r["reason"] for r in rows)))
    print("by store: ", dict(Counter(r["store"] for r in rows).most_common(10)))
    for r in rows:
        print(f"\n{datetime.fromtimestamp(r['at']):%Y-%m-%d %H:%M}  {r['reason']:<13} {r['store']} | {r['price']} | {r['title'][:80]}")
        print(f"    asked: {r['request'][:100]}")
        print(f"    {r['url']}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
