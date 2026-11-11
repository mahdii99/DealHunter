from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class Product:
    store: str
    title: str
    price: float  # EGP
    url: str
    image: Optional[str] = None
    old_price: Optional[float] = None
    rating: Optional[float] = None
    score: float = 0.0  # relevance score, filled by ranking
    note: Optional[str] = None  # LLM's short Arabic comment
    trusted: Optional[bool] = None  # well-known store?
    trust_score: Optional[int] = None  # 0-100 from signals on the store's own page (app.trust.page_signals)
    trust_signals: list = field(default_factory=list)  # Arabic labels of what was found, e.g. "رقم تليفون مصري"
    discount_pct: Optional[int] = None  # real discount vs. the store's old price / coupon
    best_deal: bool = False
    unconfirmed: bool = False  # the judge couldn't confirm the exact variant/model from the listing
    in_stock: Optional[bool] = None  # False = the store says sold out; None = couldn't tell
    verified: Optional[bool] = None  # True = its own page, opened just before showing it, says in stock at this
    # price; False = that check couldn't confirm it; None = not checked
    condition: Optional[str] = None  # 'used' / 'refurbished' / 'open box' when the store data says so
    alternative: bool = False  # not an exact match, offered because nothing exact was found
    coupon_price: Optional[float] = None  # lower price after a store coupon applied at checkout

    def to_dict(self):
        return asdict(self)


@dataclass
class ParsedQuery:
    raw: str
    keywords: list[str]               # English search terms sent to the stores
    must_have: list[str] = field(default_factory=list)  # tokens that must appear in title, e.g. "10000mah"
    max_price: Optional[float] = None
    min_price: Optional[float] = None

    @property
    def search_text(self) -> str:
        return " ".join(self.keywords + self.must_have)
