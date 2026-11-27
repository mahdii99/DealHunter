import asyncio, json, sys, time
from app import agent, catalog, db, market
from app.models import Product
NEW = sys.argv[1].split(",")
async def main():
    t = time.time()
    rep = await catalog.scrape(NEW, pages=2)
    print(json.dumps(rep, indent=1, ensure_ascii=False)); print("scrape seconds", round(time.time() - t))
    rows = db.catalog_since(market.current().code, time.time() - 3600)
    for d in NEW:
        mine = [r for r in rows if catalog.trust.domain(r["url"]) == d]
        disc = [r for r in mine if r.get("old_price") and r["old_price"] > r["price"] * 1.15]
        pick = (disc or mine)[:6]
        ps = [Product(store=r["store"], title=r["title"], price=r["price"], url=r["url"], trusted=True, trust_score=100) for r in pick]
        shown, unchecked, gone = await agent.checked_only(ps, patient=True, use_llm=False) if ps else ([], [], 0)
        print(f"{d}: {len(mine)} products, {len(disc)} with 15%+ store discount; check {len(ps)}: {len(shown)} confirmed, {len(unchecked)} unconfirmed, {gone} sold out")
        for p in shown[:2]: print("   ok", p.price, p.title[:60], p.url[:90])
        for u in unchecked[:2]: print("   no", u["url"][:100])
asyncio.run(main())
