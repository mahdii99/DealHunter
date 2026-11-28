"""The deals page a shopper browses: a price bar every card shows, filters inside a category, and a search box."""
import time

from app import deals, main

DAY = 86400


def _deal(title, cat="phones", price=100, was=150, hist=None, **kw):
    return {"title": title, "store": "Jumia", "url": f"https://www.jumia.com.eg/{abs(hash(title))}.html", "image": None,
            "price": price, "was": was, "pct": round((was - price) / was * 100), "kind": "store", "category": cat,
            "hist": hist or [], **kw}


def test_cards_are_simple_and_the_price_history_opens_on_a_tap():
    now = time.time()
    html = main._deals_html({"checked_at": now, "deals": [
        _deal("A", price=90, was=150, hist=[[now - 20 * DAY, 120], [now, 90]], real=True), _deal("B")]}, "x")
    assert html.count('class="hbtn"') == 1  # only a product we have a history for
    assert 'id="hs"' in html and "/api/history?url=" in html and "✓ خصم حقيقي" in html
    assert 'class="meter"' not in html and 'id="off"' not in html and 'id="min"' not in html


def test_brand_and_kind_come_from_the_title_with_no_llm():
    rows = [_deal("Xiaomi Redmi Note 14 Smartphone"), _deal("Apple iPhone 16 Pro 256GB"),
            _deal("Fabyo Men Winter Jacket", "fashion"), _deal("Fabyo Kids Jacket", "fashion"),
            _deal("Tornado Washing Machine 7 kg", "washers"), _deal("Nice Thing", brand="Acme", kind_ar="هدية")]
    got = deals.facets(rows)
    assert [f["brand_name"] for f in got] == ["Xiaomi", "Apple", "Fabyo", "Fabyo", "Tornado", "Acme"]
    assert got[0]["kind_name"] == "موبايلات" and got[4]["kind_name"] == "غسالات" and got[5]["kind_name"] == "هدية"


def test_search_finds_english_titles_by_the_arabic_words_shoppers_type():
    d = _deal("Samsung Galaxy A56 5G Smartphone")
    q = deals.search_text(d, deals.facets([d])[0], "موبايلات")
    assert "سامسونج" in q and "موبايل" in q and "ايفون" not in q  # an iPhone word never finds a Samsung
    w = _deal("Tornado Washing Machine 7 kg", "washers")
    assert "غساله" in deals.search_text(w, deals.facets([w])[0], "غسالات")


def test_page_has_search_sort_and_store_filters():
    html = main._deals_html({"checked_at": time.time(), "deals": [
        _deal("Samsung Galaxy A56"), _deal("LG 43 inch Smart TV", "tvs", 9000, 12000)]}, "https://x.test")
    for bit in ('id="q"', 'id="sort"', 'id="store"', 'id="sub"', 'data-brand="samsung"',
                'data-price="9000"', 'data-save="3000"', "توفّر 3,000 ج.م"):
        assert bit in html, bit
    empty = main._deals_html({"checked_at": None, "deals": []}, "https://x.test")
    assert 'id="q"' not in empty and "لسه مفيش عروض" in empty
