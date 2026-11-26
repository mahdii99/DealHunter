import asyncio

from app import agent, buybox, llm, websearch
from app.models import Product

URL = "https://shop.example.eg/product/sharp-washer-8kg"


def page(box: str, related: str = "") -> str:
    """A product page with no product data: a heading, its buy box, then related products."""
    cards = "".join(f'<li><a href="/product/other-washer-{i}">Other washer {i}</a><span class="price">{9000 + i} EGP</span></li>'
                    for i in range(4))
    return (f'<html><body><nav><a href="/">Home</a></nav><main><div class="product-main">'
            f'<div class="gallery"><img src="/a.jpg"></div><div class="summary"><h1>Sharp Washing Machine 8 Kg Inverter</h1>'
            f'{box}</div></div><section class="related"><h2>Related products</h2><ul>{related or cards}</ul></section>'
            f'</main></body></html>')


def test_the_one_price_beside_the_heading_is_read():
    html = page('<p class="price"><del>17,999 EGP</del> <ins>16,499 EGP</ins></p><button>Add to cart</button>')
    assert buybox.read(html, URL) == ("Sharp Washing Machine 8 Kg Inverter", 16499.0, True)
    # a price written for machines is read as it is ("1698.999" is not 1,698,999)
    html = page('<span data-price-amount="16499.00"><span class="price">16,499 ج.م</span></span><button>أضف للسلة</button>')
    assert buybox.read(html, URL)[1] == 16499.0


def test_sold_out_and_unclear_stock():
    assert buybox.read(page('<span class="price">16,499 EGP</span><p>Out of stock</p>'), URL)[2] is False
    # a hidden "sold out" label beside a working buy button: not known
    assert buybox.read(page('<span class="price">16,499 EGP</span><span class="label">نفذت الكمية</span>'
                            '<button>Add to cart</button>'), URL)[2] is None
    assert buybox.read(page('<span class="price">16,499 EGP</span>'), URL)[2] is None


def test_unclear_buy_boxes_are_left_to_the_llm():
    # installments and savings aren't the price; the one price left is
    html = page('<span class="price">16,499 EGP</span><p>or 1,375 EGP / month</p><p>وفر 1,500 ج.م</p>'
                '<button>Add to cart</button>')
    assert buybox.read(html, URL)[1] == 16499.0
    # two current prices (a bundle, a second offer): not clear
    assert buybox.read(page('<p>16,499 EGP</p><p>With stand 17,299 EGP</p><button>Add to cart</button>'), URL) is None
    # the price elements are all named like old prices ("price-before-discount"): can't tell them apart
    assert buybox.read(page('<div class="price-before-discount"><span class="price-value">1,599</span> ج.م</div>'
                            '<div class="strike-price"><span class="price-value">2,029</span> ج.م</div>'
                            '<button>Add to cart</button>'), URL) is None
    # no price by the heading: the related products' prices are never taken
    assert buybox.read(page('<button>Order on WhatsApp</button>'), URL) is None


def test_numbers_of_the_name_and_words_are_not_prices():
    html = page('<p class="info"><span>8 كجم</span> Apple LE 128</p><span class="price">16,499 EGP</span>'
                '<button>Add to cart</button>').replace("8 Kg", "8 جم")
    assert buybox.read(html, URL)[1] == 16499.0


def test_a_product_page_is_checked_by_its_buy_box_without_the_llm(monkeypatch):
    html = page('<p class="price">16,499 EGP</p><button>Add to cart</button>')

    async def fake_fetch(client, sem, u):
        return [], html

    async def ask_must_not_run(*a, **k):
        raise AssertionError("the buy box is clear")
    monkeypatch.setattr(agent, "VERIFY_PICKS", True)
    monkeypatch.setattr(agent, "_fetch_page", fake_fetch)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", ask_must_not_run)
    agent.new_search()
    picks = [Product("Shop", "Sharp Washing Machine 8 Kg Inverter", 16999, URL)]
    shown, _ = asyncio.run(agent._verify_stock(picks))
    # its price moved on the page: the page's price is taken
    assert [(p.price, p.verified, p.in_stock) for p in shown] == [(16499.0, True, True)]
    assert agent.page_reads()["llm"] == 0 and agent.page_reads()["without_llm"] >= 1


def test_the_web_round_reads_product_pages_by_their_buy_box(monkeypatch):
    html = page('<p class="price">16,499 EGP</p><button>Add to cart</button>')

    async def fake_search(client, query, max_results=8, exclude_domains=None, include_domains=None):
        return [{"url": URL, "title": "Sharp washer", "snippet": "", "content": ""}]

    async def fake_fetch(client, sem, u):
        return [], html

    async def ask_must_not_run(*a, **k):
        raise AssertionError("the buy box is clear")
    monkeypatch.setattr(websearch, "search", fake_search)
    monkeypatch.setattr(agent, "_fetch_page", fake_fetch)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", ask_must_not_run)
    products, _ = asyncio.run(agent._web_find(["Sharp washer 8 kg"], famous=["amazon.eg"]))
    assert [(p.title, p.price, p.in_stock) for p in products] == [("Sharp Washing Machine 8 Kg Inverter", 16499.0, True)]
