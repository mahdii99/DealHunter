"""Fixes from the live check of 2026-11-26: what a shopper saw that was wrong or confusing."""
from fastapi.testclient import TestClient

from app import agent, main, noon_index
from app.query import english_words, lacks_wanted


def test_size_or_color_badge_only_for_what_the_shopper_wrote():
    said = "تكييف شارب 1.5 حصان بارد ساخن"
    assert not agent.shopper_named("لون أبيض", said)
    assert not agent.shopper_named("موديل AY-A12ZSE", said)
    assert agent.shopper_named("1.5 حصان", said)
    said = "تيشيرت مقاس XL لون اسود"
    assert agent.shopper_named("مقاس XL", said) and agent.shopper_named("لون أسود", said)
    assert not agent.shopper_named("مقاس 46", said)


def test_cool_and_heat_request_never_shows_a_cool_only_air_conditioner():
    eng = english_words("تكييف شارب 1.5 حصان بارد ساخن")
    assert "heat" in eng.split()
    assert lacks_wanted("Sharp Split Air Conditioner 1.5 HP Cooling Only, Digital", eng)
    assert not lacks_wanted("Sharp Split Air Conditioner 1.5 HP Cool - Heat", eng)
    assert not lacks_wanted("Sharp Split Air Conditioner 1.5 HP Cooling Only", english_words("تكييف شارب 1.5 حصان"))


def test_a_need_shows_the_suggested_products_not_the_cheapest_of_the_kind():
    plan = {"need_based": True, "suggested_products": ["Samsung Galaxy A26 5G", "Honor X8c"]}
    assert agent._suggested("Samsung Galaxy A26 5G 256GB Black", plan)
    assert agent._suggested("HONOR X8c 512GB", plan)
    assert not agent._suggested("Infinix Smart 10 128GB", plan)
    cheap = agent.Product(store="Jumia", title="Infinix Smart 10 Mobile Phone 128GB", price=8599, url="a")
    good = agent.Product(store="Jumia", title="Samsung Galaxy A26 5G Mobile Phone 256GB", price=11999, url="b")
    got = agent._shelf_firsts([cheap, good], "موبايل كويس للتصوير", plan)
    assert [p.url for p in got] == ["b"]


def test_noon_names_keep_decimal_sizes():
    assert noon_index.title("sharp-split-air-conditioner-1-5-hp-cool") == "Sharp Split Air Conditioner 1.5 Hp Cool"
    assert noon_index.title("galaxy-a36-5g-8gb-ram-256gb").startswith("Galaxy A36 5G 8GB")


def test_terms_page():
    r = TestClient(main.app).get("/terms")
    assert r.status_code == 200 and "شروط الاستخدام" in r.text


def test_searches_at_the_same_time_keep_their_own_stage_times():
    import asyncio

    async def one(name: str, clear: bool):
        agent.new_search()
        agent.TIMES["search"] = 1.0
        await asyncio.sleep(0.01)
        if clear:
            agent.TIMES.clear()  # (the quick answer starts over its own times)
        await asyncio.sleep(0.02)
        return agent.TIMES["search"]

    async def both():
        return await asyncio.gather(asyncio.create_task(one("a", False)), asyncio.create_task(one("b", True)))

    assert asyncio.run(both()) == [1.0, 0.0]  # the other search's clear() never reaches this one


def test_and_joined_words_are_read_too():
    eng = english_words("تكييف شارب 1.5 حصان بارد وساخن")
    assert "heat" in eng.split()
    assert lacks_wanted("Sharp Split Air Conditioner 1.5 HP 12000 (BTU/H) Cooling Only, Digital", eng)
