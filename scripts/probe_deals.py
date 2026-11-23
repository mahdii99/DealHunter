"""Temporary: which deal listing pages our crawler may read and how many products each gives."""
import asyncio, httpx
from app import catalog, fetchers
URLS = [
 "https://www.jumia.com.eg/home-office/?page=1",
 "https://www.jumia.com.eg/health-beauty/?page=1",
 "https://www.jumia.com.eg/category-fashion-by-jumia/?page=1",
 "https://www.jumia.com.eg/electronics/?page=1",
 "https://www.jumia.com.eg/flash-sales/?page=2",
 "https://www.jumia.com.eg/home-kitchen/?page=1",
 "https://www.amazon.eg/s?i=kitchen&rh=p_8%3A25-&language=en_AE&page=1",
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
            await asyncio.sleep(10)
asyncio.run(main())
