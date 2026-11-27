"""Train the learned price reader (app/pricemodel.py) from saved store pages whose price is known.

Usage: python -m scripts.train_price_model [<folder of saved pages> [labels.json]] [--db] [--save-base] [--only-if-safe]

Always takes scripts/price_training_base.json: the facts of the saved pages it first learned from (numbers and their
facts only, no page content), so the weekly run (.github/workflows/train-price-model.yml) needs no saved pages.
--save-base rewrites that file from the folder's pages.
--db also takes the examples the agent collected live (table price_examples: every page whose price an LLM read or a
clear read knew, from every store it opened; needs DATABASE_URL). That's how the reader learns new stores.
--only-if-safe writes the new weights only when it was sure and wrong on no page of a store it hadn't seen.

A page's known price comes from its own product data (JSON-LD / og:price), or from labels.json ({file name: price},
prices read by the LLM or by clear reads). Every candidate showing that number is a positive example, the rest are
negatives. It reports how often the model reads the right price on stores it has never seen (pages split by store,
5 folds), how often it is sure, and how often it is sure and wrong; then writes app/price_model.json.
Needs scikit-learn (offline only, not on the server).
"""
import glob
import json
import os
import re
import sys
import zlib

from sklearn.linear_model import LogisticRegression

from app import pricemodel


def truth(html: str) -> set[float]:
    found = set()
    for block in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', html, re.S | re.I):
        if re.search(r'"@type"\s*:\s*\[?\s*"Product"', block):
            for v in re.findall(r'"(?:price|lowPrice)"\s*:\s*"?([\d.,]+)', block):
                try:
                    found.add(round(float(v.replace(",", "")), 2))
                except ValueError:
                    pass
    for v in re.findall(r'(?:product|og):price:amount"\s+content="([\d.,]+)', html):
        try:
            found.add(round(float(v.replace(",", "")), 2))
        except ValueError:
            pass
    return found


def pages(folder: str, labels: dict):
    for path in sorted(glob.glob(os.path.join(folder, "*.html"))):
        name = os.path.basename(path)
        html = open(path, encoding="utf-8", errors="ignore").read()
        known = {round(float(labels[name]), 2)} if name in labels else truth(html)
        if len(known) != 1:  # no known price, or a range / several: not a clear example
            continue
        url = "https://" + name[:-5] + "/product"
        cands = pricemodel.candidates(html, url)
        price = known.pop()
        if not any(abs(v - price) < 1 for v, _ in cands):
            continue  # the price isn't shown as a price element: nothing to learn from this page
        yield name, price, cands


BASE_FILE = os.path.join(os.path.dirname(__file__), "price_training_base.json")


def _store(row) -> str:
    from app import trust
    name = row[0]
    return trust.domain(name if name.startswith("http") else "https://" + name.removesuffix(".html"))


def base():
    try:
        with open(BASE_FILE, encoding="utf-8") as f:
            return [(name, float(price), [(float(v), feats) for v, feats in cands]) for name, price, cands in json.load(f)]
    except OSError:
        return []


def collected():
    """The examples the agent saved live (app/pricemodel.remember)."""
    from app import db
    for row in db.price_examples():
        cands = [(float(v), f) for v, f in json.loads(row["candidates"])]
        if any(abs(v - row["price"]) < 1 for v, _ in cands):
            yield row["url"], float(row["price"]), cands


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = args[0] if args else None
    labels = json.load(open(args[1])) if len(args) > 1 else {}
    if folder:
        data = list(pages(folder, labels))
        if "--save-base" in sys.argv:
            with open(BASE_FILE, "w", encoding="utf-8") as f:
                json.dump([[n, p, [[v, {k: round(x, 3) for k, x in feats.items()}] for v, feats in c]] for n, p, c in data],
                          f, separators=(",", ":"))
            print("wrote", BASE_FILE)
    else:
        data = base()
    print(f"{len(data)} saved pages")
    if "--db" in sys.argv:
        from app import db
        live = list(collected())
        if os.getenv("DATABASE_URL") and not db._db.pg:
            sys.exit("can't reach DATABASE_URL: not training without the live examples")
        print(f"{len(live)} examples collected live")
        data += live
    # (folds split by store: a store's pages are all in one fold, so every test is on stores the model never saw)
    keys = sorted({k for _, _, cands in data for _, f in cands for k in f})
    print(f"{len(data)} pages, {sum(len(c) for _, _, c in data)} candidates, features: {len(keys)}")

    def xy(rows):
        X, y = [], []
        for _, price, cands in rows:
            for v, f in cands:
                X.append([f.get(k, 0.0) for k in keys])
                y.append(1 if abs(v - price) < 1 else 0)
        return X, y

    def fit(rows):
        X, y = xy(rows)
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        m.fit(X, y)
        return {"bias": float(m.intercept_[0]), **{k: float(c) for k, c in zip(keys, m.coef_[0])}}

    folds = 5
    right = sure = wrong = 0
    sweep = {t: [0, 0] for t in (0.8, 0.9, 0.95, 0.97, 0.99)}
    for k in range(folds):
        fold = lambda r: zlib.crc32(_store(r).encode()) % folds  # noqa: E731  (the same split on every run)
        train = [r for r in data if fold(r) != k]
        test = [r for r in data if fold(r) == k]
        w = fit(train)
        for name, price, cands in test:
            scored = sorted(((pricemodel.score(f, w), v) for v, f in cands), reverse=True)
            best_p, best_v = scored[0]
            other = max((p for p, v in scored[1:] if abs(v - best_v) >= 1), default=0.0)
            ok = abs(best_v - price) < 1
            right += ok
            for t in sweep:
                if best_p >= t and best_p - other >= pricemodel.MARGIN:
                    sweep[t][0] += 1
                    sweep[t][1] += not ok
            if best_p >= pricemodel.MIN_SURE and best_p - other >= pricemodel.MARGIN:
                sure += 1
                if not ok:
                    wrong += 1
                    print("  sure and wrong:", name, best_v, "(the price is", price, ")")
    n = len(data)
    print(f"unseen stores: top pick right {right}/{n}; sure {sure}/{n}; sure and wrong {wrong}")
    print("  sure at each threshold (pages, wrong):", sweep)
    if "--only-if-safe" in sys.argv and wrong:
        sys.exit("sure and wrong on unseen stores: keeping the current weights")
    w = fit(data)
    with open(pricemodel.WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump({k: round(v, 4) for k, v in w.items()}, f, indent=0, sort_keys=True)
    print("wrote", pricemodel.WEIGHTS_FILE)


if __name__ == "__main__":
    main()
