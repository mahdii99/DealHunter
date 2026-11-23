"""Temporary: which deal listing pages our crawler may read and how many products each gives."""
import asyncio, httpx
from app import catalog, fetchers
URLS = [
 "https://www.amazon.eg/s?i=electronics&rh=p_8%3A30-&language=en_AE",
 "https://www.amazon.eg/s?i=kitchen&rh=p_8%3A30-&language=en_AE",
 "https://www.amazon.eg/s?i=fashion&rh=p_8%3A30-&language=en_AE",
 "https://www.amazon.eg/s?i=beauty&rh=p_8%3A30-&language=en_AE",
 "https://www.amazon.eg/s?k=deals&pct-off=30-&language=en_AE",
 "https://www.amazon.eg/s?i=home-garden&rh=p_8%3A30-&language=en_AE",
 "https://www.jumia.com.eg/flash-sales/",
 "https://www.jumia.com.eg/mlp-deals/",
 "https://www.jumia.com.eg/home-office/?price_discount=30-100",
 "https://www.jumia.com.eg/health-beauty/?price_discount=30-100",
 "https://www.jumia.com.eg/electronics/?price_discount=30-100",
 "https://www.jumia.com.eg/category-fashion-by-jumia/?price_discount=30-100",
 "https://www.jumia.com.eg/phones-tablets/?price_discount=30-100",
 "https://www.jumia.com.eg/computing/?price_discount=30-100",
 "https://www.jumia.com.eg/home-office/?sort=highest-discount",
]
async def main():
    async with httpx.AsyncClient() as c:
        for u in URLS:
            try:
                ps = await catalog._page(c, u)
                off = [round((p.old_price - p.price) / p.old_price * 100) for p in ps if p.old_price and p.old_price > p.price]
                print("OK", len(ps), "with old price:", len(off), "min off:", min(off) if off else None, u, "|", ps[0].title[:50] if ps else "")
            except Exception as e:
                print("FAIL", type(e).__name__, str(e)[:120], u)
            await asyncio.sleep(4)
asyncio.run(main())
