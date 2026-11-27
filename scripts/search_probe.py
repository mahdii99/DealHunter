"""Temporary: how a keyless search service answers shoppers' requests from a datacenter IP."""
import json, time, urllib.request
QS = ["Samsung Galaxy A55 price Egypt", "سعر سامسونج جالاكسي A55", "غسالة توشيبا 8 كيلو فوق اتوماتيك",
      "شاشة سامسونج 55 بوصة 4K", "iPhone 15 128GB Egypt price", "لابتوب لينوفو ideapad i5 سعر",
      "ثلاجة شارب 18 قدم", "سماعة JBL Tune 520", "Xiaomi Redmi Note 13 site:jumia.com.eg",
      "Samsung A55 (site:amazon.eg OR site:jumia.com.eg OR site:btech.com OR site:2b.com.eg)"]
def post(url, body, h):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **h})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300], time.time() - t
    except Exception as e:
        return 0, repr(e), time.time() - t
for q in QS:
    code, d, s = post("https://api.keenable.ai/v1/search/public", {"query": q, "max_results": 10},
                      {"X-Keenable-Title": "DealHunter"})
    print(f"\n### keenable {code} {s:.1f}s  {q}")
    if code == 200:
        for r in d.get("results", []):
            print("  ", r.get("url", "")[:110], "|", (r.get("title") or "")[:60], "|", (r.get("snippet") or "")[:90].replace("\n", " "))
    else:
        print("  ", d)
    code, d, s = post("https://api.keenable.ai/v1/search/public", {"query": q, "max_results": 10,
                      "exclude_domains": ["amazon.eg", "jumia.com.eg", "noon.com", "btech.com"]}, {"X-Keenable-Title": "DealHunter"})
    print(f"  -- excluding big stores: {code} {s:.1f}s", [r.get("url", "")[:70] for r in (d.get("results", []) if code == 200 else [])])
