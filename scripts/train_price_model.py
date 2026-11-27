"""Train the learned price reader (app/pricemodel.py) from saved store pages whose price is known.

Usage: python -m scripts.train_price_model <folder of saved pages> [labels.json] [--db]

--db also takes the examples the agent collected live (table price_examples: every page whose price an LLM read or a
clear read knew, from every store it opened; needs DATABASE_URL). That's how the reader learns new stores.

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


def _store(row) -> str:
    from app import trust
    return trust.domain(row[0] if row[0].startswith("http") else "https://" + row[0][:-5])


def collected():
    """The examples the agent saved live (app/pricemodel.remember)."""
    from app import db
    for row in db.price_examples():
        cands = [(float(v), f) for v, f in json.loads(row["candidates"])]
        if any(abs(v - row["price"]) < 1 for v, _ in cands):
            yield row["url"], float(row["price"]), cands


def main():
    args = [a for a in sys.argv[1:] if a != "--db"]
    folder = args[0]
    labels = json.load(open(args[1])) if len(args) > 1 else {}
    data = list(pages(folder, labels))
    if "--db" in sys.argv:
        live = list(collected())
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
        fold = lambda r: hash(_store(r)) % folds  # noqa: E731
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
    w = fit(data)
    with open(pricemodel.WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump({k: round(v, 4) for k, v in w.items()}, f, indent=0, sort_keys=True)
    print("wrote", pricemodel.WEIGHTS_FILE)


if __name__ == "__main__":
    main()
