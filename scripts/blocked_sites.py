"""Sites the agent skips because their pages never opened (bot protection, JavaScript-only, robots.txt...).

    python -m scripts.blocked_sites                    # list them
    python -m scripts.blocked_sites --unblock zara.com # try a site again (e.g. after we found a way to read it)
"""
import sys
from datetime import datetime

from app import db


def main(args: list[str]) -> None:
    if len(args) == 2 and args[0] == "--unblock":
        print("unblocked" if db.unblock(args[1].lower().removeprefix("www.")) else "not in the list", args[1])
        return
    rows = db.list_blocked()
    if not rows:
        print("no blocked sites")
    for r in rows:
        state = "SKIPPED " if r["blocked"] else f"failed {r['fails']}x"
        print(f"  {state:<9} {r['domain']:<30} {datetime.fromtimestamp(r['last_failed']):%Y-%m-%d %H:%M}  {r['reason'][:80]}")


if __name__ == "__main__":
    main(sys.argv[1:])
