"""Beauty and personal-care requests (perfume, makeup, skin and hair care, hair tools), the products many shoppers buy
online most. Live 2026-11-27 with no model: "برفان لانكوم لافي ايبل 100 مل", "سيروم فيتامين سي للبشرة" and
"روج ميبلين سوبر ستاي" found nothing (the words weren't understood), "شامبو بانتين 400 مل" showed a Kamena kids
shampoo, and "مكواة شعر فيليبس" showed Philips steam irons for clothes."""
import json

import pytest

from app import agent, catalog, db, llm, trust
from app.query import (copy_or_tester, english_words, is_accessory, is_beauty, kinds_in, names_brand, other_gender,
                       parse_query, shows_size, wanted_sizes)
from app.stores.shopify import parse_suggest

P = agent.Product


@pytest.fixture(autouse=True)
def _own_database(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "STATE_FILE", tmp_path / "llm_state.json")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    yield
    db.close()


def _why(eng: str, *titles: str) -> list:
    return [agent._why_not(P("X", t, 100, f"https://x/{i}"), eng) for i, t in enumerate(titles)]


def test_beauty_requests_understood_by_words():
    assert english_words("برفان لانكوم لافي ايبل 100 مل") == "perfume lancome la vie est belle 100 ml"
    assert english_words("شامبو بانتين 400 مل") == "shampoo pantene 400 ml"
    assert english_words("روج ميبلين سوبر ستاي") == "lipstick maybelline superstay"
    assert english_words("مكواة شعر فيليبس") == "hair straightener philips"
    assert english_words("سيروم فيتامين سي للبشرة الدهنية") == "serum vitamin c oily skin"
    assert english_words("شامبو هيد اند شولدرز ضد القشرة") == "shampoo head-shoulders anti dandruff"
    assert english_words("كريم لوريال 50 جرام") == "cream loreal 50 gram"
    assert wanted_sizes(english_words("perfume 100ml")) == [("100", "ml")]


def test_millilitres_are_mah_only_for_batteries():
    assert parse_query("برفان 100 مللي").must_have == ["100ml"]
    assert parse_query("باور بانك 10 الاف مللي").must_have == ["10000mah"]
    assert "ml" not in english_words("باور بانك انكر 10000 مللي")


def test_sizes_in_ml_and_grams():
    assert shows_size("Lancome La Vie Est Belle EDP 100ml", "100", "ml")
    assert shows_size("شامبو بانتين 400 مل", "400", "ml")
    assert not shows_size("Lancome La Vie Est Belle EDP 100ml", "10", "ml")
    assert not shows_size("Eyeliner 10 ملم", "10", "ml")  # millimetres
    assert shows_size("Nivea Creme 50 g", "50", "gram") and not shows_size("Phone 50GB", "50", "gram")


def test_kinds_of_beauty_products():
    assert kinds_in("Pantene Pro-V Conditioner 360ml") == {"hair conditioner"}
    assert kinds_in("Sharp Split Air Conditioner 1.5 HP") == {"air conditioner"}  # not hair conditioner
    assert kinds_in("Sokany Rotating Curly Hair Iron 32mm") == {"hair straightener"}  # not a clothes iron
    assert kinds_in("Philips Steam Iron 2000 Watt") == {"iron"}
    assert kinds_in("Lancome Idole Eau de Parfum 50ml") == {"perfume"}


def test_brands_written_with_accents_and_symbols():
    assert names_brand("L’Oréal Paris Revitalift Serum 30ml", "loreal")
    assert names_brand("Head & Shoulders Classic Clean Shampoo", "head-shoulders")
    assert names_brand("La Roche-Posay Effaclar Duo+", "la-roche-posay")
    assert not names_brand("Kamena Kids Shampoo 400 ml", "pantene")


def test_skin_care_is_not_a_phone_skin():
    assert not is_accessory("Vitamin C Serum for all skin types 30ml")
    assert not is_accessory("Maybelline Fit Me Full Cover Concealer")
    assert not is_accessory("Makeup brush kit 12 pcs")
    assert not is_accessory("BaByliss hair dryer with diffuser and stand")
    assert is_accessory("iPhone 15 skins sticker wrap")
    assert is_accessory("Phone case for Samsung A15")


def test_what_live_searches_showed_is_left_out():
    assert _why("shampoo pantene 400 ml", "Kamena Kids Shampoo with Soft Scent - 400 ml",
                "Pantene Pro-V Shampoo Classic Care 400ml", "Pantene Shampoo 200ml") == ["brand", None, "size"]
    assert _why("hair straightener philips", "Philips 2000 Series Steam Iron , 2000 Watt",
                "Philips Hair Straightener BHS376 ThermoProtect") == ["kind", None]


def test_copies_testers_and_the_other_gender():
    eng = english_words("برفان لانكوم حريمي")
    assert _why(eng, "Lattafa perfume inspired by Lancome La Vie Est Belle", "Lancome Idole EDP Tester 50ml",
                "Lancome Hypnose Homme EDT for Men", "Lancome La Vie Est Belle EDP for Women 100ml") == \
        ["copy", "tester", "gender", None]
    assert copy_or_tester("Lancome Idole Tester", "perfume tester lancome") is None
    assert not other_gender("Dior Sauvage EDT for Men", "perfume dior men")
    assert not other_gender("Lattafa Oud unisex", "perfume women")


def test_store_titles_say_perfume_other_ways():
    # "Eau de Parfum" / "EDP" is the perfume the shopper wrote as "برفان"
    got = agent._fitting([P("Amazon.eg", "Lancome La Vie Est Belle Eau de Parfum 100ml", 5000, "https://a/1"),
                          P("Amazon.eg", "Lancome La Vie Est Belle EDP 50ml", 4000, "https://a/2")],
                         english_words("برفان لانكوم لافي ايبل"))
    assert len(got) == 2


def test_price_list_finds_beauty_products_by_their_arabic_words():
    db.save_catalog([
        {"url": "https://www.jumia.com.eg/lancome.html", "store": "Jumia", "price": 5200,
         "title": "Lancome La Vie Est Belle Eau De Parfum For Women - 100ml"},
        {"url": "https://www.jumia.com.eg/kamena.html", "store": "Jumia", "price": 74,
         "title": "Kamena Kids Shampoo with Soft Scent - 400 ml"},
        {"url": "https://www.jumia.com.eg/pantene.html", "store": "Jumia", "price": 150,
         "title": "Pantene Pro-V Shampoo Classic Care 400ml"},
    ], "EG", "perfumes")
    got = agent._shelf_offers("برفان لانكوم لافي ايبل 100 مل", "برفان لانكوم لافي ايبل 100 مل")
    assert [p.price for p in got] == [5200]
    got = [p for p in agent._shelf_offers("شامبو بانتين 400 مل", "شامبو بانتين 400 مل")
           if agent._why_not(p, english_words("شامبو بانتين 400 مل")) is None]
    assert [p.price for p in got] == [150]


def test_beauty_shops_searched_for_beauty_requests_only():
    assert is_beauty("perfume lancome") and is_beauty("hair dryer babyliss") and is_beauty("serum vitamin c")
    assert not is_beauty("tv samsung 55 inch") and not is_beauty("phone realme 12 pro")
    assert trust.is_known("https://sourcebeauty.com/products/x")
    assert "sourcebeauty.com" not in trust.FAMOUS_SITES  # not searched for phones


def test_shopify_search_answer_read_without_llm():
    text = json.dumps({"resources": {"results": {"products": [
        {"available": True, "title": "Glow Deep Serum", "vendor": "Beauty of Joseon", "price": "1020.00",
         "compare_at_price_max": "1200.00", "url": "/products/glow-deep-serum?_pos=1", "image": "https://cdn/x.png"},
        {"available": False, "title": "Sold out serum", "vendor": "X", "price": "300.00", "url": "/products/y"},
        {"title": "No price", "url": "/products/z", "price": ""}]}}})
    got = parse_suggest(text, "https://sourcebeauty.com/search/suggest.json?q=serum", "Source Beauty")
    assert [(p.title, p.price, p.old_price, p.url, p.in_stock) for p in got] == [
        ("Beauty of Joseon Glow Deep Serum", 1020.0, 1200.0, "https://sourcebeauty.com/products/glow-deep-serum", True),
        ("X Sold out serum", 300.0, None, "https://sourcebeauty.com/products/y", False)]
    assert parse_suggest("<html>", "https://x/search", "X") == []


def test_beauty_sections_in_our_price_list():
    for cat in ("perfumes", "makeup", "skincare", "haircare", "hair_tools"):
        assert cat in catalog.CATEGORIES and catalog.CATEGORIES[cat]["mixed"]


def test_a_perfumes_other_editions_are_other_products():
    # live 2026-11-27 with a model: "برفان لانكوم لافي ايبل 100 مل" also showed its Intensément, L'Elixir, Iris
    # Absolu and a refill bottle
    eng = english_words("برفان لانكوم لافي ايبل 100 مل")
    assert _why(eng, "Lancome la vie est belle - perfumes for women - eau de parfum, 100 ml",
                "LANCOME LAVIE EST BELLE EDP REFILL 100ML", "Lancôme La Vie Est Belle Iris Absolu EDP For Her 100ml",
                "La Vie Est Belle L'Elixir Recharge For Her EDP 100ml",
                "Lancome la vie est belle intensement for women 100ml") == [None, "edition", "edition", "brand",
                                                                             "edition"]
    assert agent._why_not(P("X", "Lancome La Vie Est Belle Intensement EDP 100ml", 1, "u"),
                          english_words("برفان لانكوم لافي ايبل intense 100 مل")) is None
