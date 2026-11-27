import asyncio, re, sys, pathlib
import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser
from app import catalog
OUT = pathlib.Path("probe_pages"); OUT.mkdir(exist_ok=True)
SEL = ["a.product-item-link", ".product-title a", "a.product-thumbnail", "a[href*='/products/']", "a[href*='/p/']"]
async def main(urls):
    async with httpx.AsyncClient(headers=catalog.BOT_HEADERS, follow_redirects=True, timeout=30) as c:
        for i, u in enumerate(urls):
            try:
                r = await c.get(u)
            except Exception as e:
                print(u, repr(e)); continue
            (OUT / f"{i:02d}.html").write_text(f"<!-- {u} -->\n" + r.text)
            print(i, u, r.status_code, len(r.text))
            if u.endswith(".json"):
                continue
            t = HTMLParser(r.text)
            for s in SEL:
                n = t.css_first(s)
                if n and n.attributes.get("href"):
                    pu = str(httpx.URL(str(r.url)).join(n.attributes["href"]))
                    await asyncio.sleep(2)
                    try:
                        pr = await c.get(pu)
                        (OUT / f"{i:02d}p.html").write_text(f"<!-- {pu} -->\n" + pr.text)
                        print("  product", pu, pr.status_code, len(pr.text))
                    except Exception as e:
                        print("  product", pu, repr(e))
                    break
            await asyncio.sleep(2)
asyncio.run(main(sys.argv[1:]))
