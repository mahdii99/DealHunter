"""Steps done with no LLM call where a rule reads the page as well: listing pages' product cards, the market check
of a named product, and a size option the product's own name shows."""
import asyncio

from app import agent, cards, llm


def _woo(name: str, slug: str, price: str, old: str = "", extra: str = "") -> str:
    old_html = f'<del><span class="woocommerce-Price-amount amount">{old}</span></del>' if old else ""
    return (f'<li class="product type-product product_cat-washing-machines">'
            f'<a href="https://shop.example.com/product/{slug}/"><img src="/img/{slug}.jpg" alt="{name}"></a>'
            f'<div class="star-rating">تم التقييم <strong class="rating">5</strong> من 5</div>'
            f'<h2 class="woocommerce-loop-product__title"><a href="/product/{slug}/">{name}</a></h2>'
            f'<span class="price">{old_html}<ins><span class="woocommerce-Price-amount amount"><bdi>'
            f'<span class="woocommerce-Price-currencySymbol">EGP</span>{price}</bdi></span></ins></span>{extra}'
            f'<a href="/product/{slug}/" class="button">Add to cart</a></li>')


WOO_PAGE = ("<html><body><nav><a href='/product-category/washers/'>Washers</a><a href='/cart/'>Cart</a></nav>"
            "<ul class='products'>"
            + _woo("Toshiba Washing Machine 10 Kg Top Load", "toshiba-10", "18,499", old="21,000")
            + _woo("Sharp Washing Machine 9 Kg Top Load", "sharp-9", "15,999")
            + _woo("Fresh Washing Machine 8 Kg Half Automatic", "fresh-8", "7,250",
                   extra='<div class="out-of-stock-label">غير متوفر في المخزون</div>')
            + "</ul><footer><a href='/privacy-policy/'>Privacy</a> 2026</footer></body></html>")


def test_listing_cards_are_read_with_their_own_name_link_and_price():
    got = cards.listing_cards(WOO_PAGE, "shop", "https://shop.example.com/search?q=washer")
    assert [(p.title, p.price, p.url) for p in got] == [
        ("Toshiba Washing Machine 10 Kg Top Load", 18499.0, "https://shop.example.com/product/toshiba-10/"),
        ("Sharp Washing Machine 9 Kg Top Load", 15999.0, "https://shop.example.com/product/sharp-9/"),
        ("Fresh Washing Machine 8 Kg Half Automatic", 7250.0, "https://shop.example.com/product/fresh-8/")]
    # the crossed-out price is the old one, never the price; a rating "5 من 5" beside the currency isn't a price;
    # a category class "product_cat-washing-machines" isn't an old-price class
    assert got[0].old_price == 21000 and got[1].old_price is None
    assert [p.in_stock for p in got] == [None, None, False]
    assert got[0].image == "https://shop.example.com/img/toshiba-10.jpg"


def test_card_prices_in_any_way_stores_write_them():
    def page(price_html: str) -> str:
        return "<div>" + "".join(
            f'<div class="card"><a href="/p/item-number-{i}">Phone model number {i}</a>{price_html}</div>'
            for i in range(3)) + "</div>"
    assert {p.price for p in cards.listing_cards(page("<p>4.500,00 EGP 3.800,00 EGP</p>".replace(
        "4.500,00 EGP", "<s>4.500,00 EGP</s>")), "s", "https://s.com/")} == {3800.0}
    assert {p.price for p in cards.listing_cards(page("<span>١٨٬٥٠٠ جنيه</span>"), "s", "https://s.com/")} == {18500.0}
    assert {p.price for p in cards.listing_cards(
        page('<span data-price-amount="21049" class="price-wrapper">21,049 جنيه</span>'
             '<span class="old-price"><span data-price-amount="23000">23,000</span></span>'),
        "s", "https://s.com/")} == {21049.0}
    # two different current prices (cash or a monthly installment): not clear, left to the LLM
    assert cards.listing_cards(page("<p>9,500 ج.م أو 1,124 ج.م شهريا</p>"), "s", "https://s.com/") == []


def test_a_page_with_few_priced_links_is_not_a_listing():
    menu = ("<nav>" + "".join(f'<a href="/category/kind-{i}/">Kind {i} products</a>' for i in range(8)) + "</nav>"
            '<div><a href="/p/the-one-product">The One Product Name</a> EGP 1,000</div>')
    assert cards.listing_cards(menu, "s", "https://s.com/") == []


def test_web_search_listing_pages_are_read_without_the_llm(monkeypatch):
    from app import websearch

    async def fake_search(client, query, max_results=8, exclude_domains=None, include_domains=None):
        return [{"url": "https://shop.example.com/search?q=washer", "title": "", "snippet": ""},
                {"url": "https://fanstore.com/fans", "title": "", "snippet": ""}]

    fans = WOO_PAGE.replace("Washing Machine", "Stand Fan").replace("10 Kg", "16 inch").replace(
        "9 Kg", "18 inch").replace("8 Kg", "20 inch").replace("Top Load", "Remote").replace(
        "Half Automatic", "Remote").replace("Toshiba", "Unionaire").replace("shop.example.com", "fanstore.com"
                                                                            ).replace("</ul>", "</ul>" + "<p>about us</p>" * 200)

    async def fetch(client, sem, url):
        return [], WOO_PAGE if "shop." in url else fans

    read = []

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, step="", **kw):
        read.append(step)
        return {"products": []}

    monkeypatch.setattr(websearch, "search", fake_search)
    monkeypatch.setattr(agent, "_fetch_page", fetch)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    products, _ = asyncio.run(agent._web_find(["غسالة توشيبا 10 كيلو", "Toshiba washing machine 10 kg"],
                                              famous=["amazon.eg"]))
    assert {p.title for p in products} >= {"Toshiba Washing Machine 10 Kg Top Load"}
    assert all("shop.example.com" in p.url for p in products)
    # the washers page needed no LLM; the fans page lists nothing about the request: the LLM still reads it
    assert read == ["read_page"]


def test_no_second_planner_call_for_a_named_product(monkeypatch):
    calls = []

    async def fake_ask(prompt, timeout=10, prefer=None, step="", **kw):
        calls.append(step)
        return {"action": "search", "market_query": "سعر ريلمي 12 برو بلس", "need_based": False,
                "request": "ريلمي 12 برو بلس 256", "product": {"kind": "smartphone", "brand": "realme"},
                "web_queries": ["ريلمي 12 برو بلس 256"]}

    async def market_must_not_run(query, plan):
        raise AssertionError("a named product's prices come from the search itself")
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    monkeypatch.setattr(agent, "_market_check", market_must_not_run)
    plan = asyncio.run(agent.plan_turn([{"role": "user", "text": "ريلمي 12 برو بلس 256"}]))
    assert calls == ["plan"] and plan["action"] == "search"


def test_a_need_still_gets_its_market_check(monkeypatch):
    calls = []

    async def fake_ask(prompt, timeout=10, prefer=None, step="", **kw):
        calls.append(step)
        return {"action": "ask", "market_query": "اسعار لابتوب RTX 4060", "need_based": True,
                "reply": "هتستخدمه في إيه؟", "options": ["ألعاب", "شغل"]}

    async def market(query, plan):
        return "- Laptop RTX 4060 | 55,000 EGP | shop.com"
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    monkeypatch.setattr(agent, "_market_check", market)
    asyncio.run(agent.plan_turn([{"role": "user", "text": "عايز لابتوب للألعاب"}]))
    assert calls == ["plan", "plan_market"]


def test_a_size_option_the_product_name_shows_needs_no_llm_read():
    page = ("<html><body><h1>Samsung Galaxy A36 5G 256GB Storage 8GB RAM Black</h1>"
            "<span class='price'>EGP 23,500</span><button>Add to cart</button></body></html>")
    title = "Samsung Galaxy A36 5G, 256GB, 8GB RAM"
    assert agent._size_option_shown("256 جيجا", title, page)
    # the page lets the shopper pick another storage: the price shown may be another size's
    assert not agent._size_option_shown("256 جيجا", title, page.replace(
        "<button>", "<div>128GB</div><div>256GB</div><button>"))
    assert not agent._size_option_shown("256 جيجا، لون أسود", title, page)  # a color: the LLM reads it
    assert not agent._size_option_shown("مقاس 46", "Nike Air shoe 46", page)  # a clothing size too
    assert not agent._size_option_shown("128 جيجا", title, page)

    p = agent.Product("Amazon.eg", title, 23500, "https://www.amazon.eg/dp/B0TEST1234")
    sold: set[int] = set()
    assert agent._settled_without_llm(p, page, sold, "256 جيجا")
    assert p.verified and p.option_ok and p.option == "256 جيجا"


def test_a_product_page_of_a_store_with_a_learned_recipe_needs_no_llm_read(monkeypatch):
    from app import recipes, websearch

    async def fake_search(client, query, max_results=8, exclude_domains=None, include_domains=None):
        return [{"url": "https://shop.example.com/p/sharp-washer-9kg-es-tn09", "title": "", "snippet": ""}]
    page = ("<html><h1>Sharp Washing Machine 9 Kg ES-TN09</h1><div class='a'>EGP 15,999</div>"
            "<div class='b'>EGP 1,333 / month</div>" + "<p>details</p>" * 300 + "</html>")

    async def fetch(client, sem, url):
        return [], page

    async def ask_must_not_run(*a, **k):
        raise AssertionError("the store's recipe reads this page")
    monkeypatch.setattr(websearch, "search", fake_search)
    monkeypatch.setattr(agent, "_fetch_page", fetch)
    monkeypatch.setattr(recipes, "read", lambda url, html: (15999.0, True))
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", ask_must_not_run)
    products, _ = asyncio.run(agent._web_find(["Sharp washer 9 kg"], famous=["amazon.eg"]))
    assert [(p.title, p.price, p.in_stock) for p in products] == [("Sharp Washing Machine 9 Kg ES-TN09", 15999.0, True)]
