"""Send a notification on the site to every shopper of one region, e.g. when searching there goes live:

    python -m scripts.announce SA "البحث في السعودية اشتغل!" "دلوقتي بندورلك في متاجر السعودية وبالريال."
"""
import sys

from app import main  # noqa: F401  (loads .env, so DATABASE_URL is used)
from app import accounts


def run(region: str, title: str, body: str) -> int:
    users = accounts.users_in(region.upper())
    for uid in users:
        accounts.notify(uid, "info", title, body)
    return len(users)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    print(f"sent to {run(*sys.argv[1:])} shoppers")
