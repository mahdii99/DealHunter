"""A product's price history, drawn and judged: the chart and the "is this discount real?" label.

Used by every page that shows a product (search results through /api/history, /deals, product pages). Everything is
server-side and returns plain text, so a page made by the server (search engines read it) shows it with no script.

Points are what db.history() / db.histories() return, oldest first: [{"price", "in_stock", "seen_at"}]. A price
holds until the next point (we save a price again only when it changes or a day later), so the chart is a step line
running to now.

The label compares the price shown now with what the same store page asked over the last 30 days, never with the
store's own "old price" alone: stores often show a "before" price the product never sold at.
"""
import time
from datetime import datetime, timedelta, timezone
from html import escape

DAYS = 30            # the window the label speaks about
MIN_DAYS = 7         # less history than this: no label (we don't know yet)
TODAY = 20 * 3600    # points this recent are "now", not history (the price we just checked)
REAL_DROP = 0.03     # 3% under the lowest price of the window = a real drop
CLAIM = 0.10         # the store says at least 10% off
NEVER_SOLD = 0.95    # ...but its highest price in the window stayed 5% under that "before" price
CAIRO = timezone(timedelta(hours=2))


def _money(n: float) -> str:
    return f"{n:,.0f}"


def window(points: list[dict], now: float | None = None, days: int = DAYS) -> tuple[list[float], int]:
    """The prices the page asked during the last `days` days, before today (the price that held when the window
    began counts too), and how many days of history that is."""
    now = now or time.time()
    start, end = now - days * 86400, now - TODAY
    past = [p for p in points if p["seen_at"] < end]
    if not past:
        return [], 0
    held = [p for p in past if p["seen_at"] < start][-1:]  # the price at the window's start
    inside = [p for p in past if p["seen_at"] >= start]
    first = start if held else inside[0]["seen_at"]
    return [p["price"] for p in held + inside], int((now - first) // 86400)


def verdict(points: list[dict], price: float | None, old_price: float | None = None,
            now: float | None = None) -> dict | None:
    """{"kind", "text", "low", "high", "days"} or None when we don't have enough history to say anything.
    kind: "real" = cheaper than any price of the last 30 days; "fake" = the store shows a discount from a price this
    page never asked in that time; "higher" = it was clearly cheaper lately; "usual" = its usual price."""
    if not price:
        return None
    prices, days = window(points, now)
    if not prices or days < MIN_DAYS:
        return None
    low, high = min(prices), max(prices)
    base = {"low": low, "high": high, "days": days}
    if price <= low * (1 - REAL_DROP):
        return {**base, "kind": "real", "text": f"خصم حقيقي: أرخص من أي سعر شفناه في آخر {days} يوم"}
    if old_price and old_price >= price * (1 + CLAIM) and high < old_price * NEVER_SOLD and price >= low * 0.97:
        return {**base, "kind": "fake",
                "text": f"الخصم مش حقيقي: ماشفناش سعره {_money(old_price)} في آخر {days} يوم، ده سعره العادي"}
    if price > low * 1.05:
        return {**base, "kind": "higher", "text": f"كان أرخص: {_money(low)} في آخر {days} يوم"}
    return {**base, "kind": "usual", "text": f"ده سعره العادي في آخر {days} يوم"}


def _day(t: float) -> str:
    d = datetime.fromtimestamp(t, CAIRO)
    return f"{d.day}/{d.month}"


STYLE = ("<style>.pc{display:block;width:100%;height:auto;overflow:visible;direction:ltr}.pc .ln{fill:none;stroke:var(--accent,#0F766E);"
         "stroke-width:2;stroke-linejoin:round}.pc .ar{fill:var(--accent,#0F766E);opacity:.10}.pc .gd{stroke:var(--border,#DCE4E1);"
         "stroke-width:1}.pc .tx{fill:var(--muted,#56665F);font:11px system-ui,sans-serif}.pc .lo{fill:var(--ok,#047857)}"
         ".pc .dt{fill:var(--accent,#0F766E);stroke:var(--card,#fff);stroke-width:2}.pc .hit{fill:transparent}</style>")


def svg(points: list[dict], now: float | None = None, days: int = 90, width: int = 320, height: int = 120,
        compact: bool = False, title: str = "تاريخ السعر", style: bool = True) -> str:
    """The price over the last `days` days as an inline SVG (a step line to now), or "" with fewer than 2 points.
    compact: a sparkline (no labels) for small cards. Colors come from the page's --accent/--border/--muted/--ok, so it
    follows the page's light and dark themes (style=False: the page already holds STYLE once). Every point has a native tooltip (date and price)."""
    now = now or time.time()
    pts = [p for p in points if p.get("price") and p["seen_at"] >= now - days * 86400]
    older = [p for p in points if p.get("price") and p["seen_at"] < now - days * 86400][-1:]
    if older:  # the price that held when the chart's time begins
        pts = [{**older[0], "seen_at": now - days * 86400}] + pts
    if len(pts) < 2:
        return ""
    t0, t1 = pts[0]["seen_at"], now
    lo, hi = min(p["price"] for p in pts), max(p["price"] for p in pts)
    pad = (hi - lo) * 0.15 or hi * 0.05 or 1
    y0, y1 = lo - pad, hi + pad
    left, right, top, bottom = (2, 2, 4, 4) if compact else (6, 56, 10, 22)
    w, h = width - left - right, height - top - bottom

    def x(t):
        return left + (t - t0) / ((t1 - t0) or 1) * w

    def y(v):
        return top + (y1 - v) / ((y1 - y0) or 1) * h

    line = [f"M{x(pts[0]['seen_at']):.1f},{y(pts[0]['price']):.1f}"]
    for a, b in zip(pts, pts[1:]):
        line.append(f"H{x(b['seen_at']):.1f}V{y(b['price']):.1f}")
    line.append(f"H{x(t1):.1f}")
    d = "".join(line)
    base = top + h
    area = f"{d}V{base:.1f}H{x(t0):.1f}Z"
    parts = [f'<svg class="pc" direction="ltr" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
             f'aria-label="{escape(title)}: من {_money(lo)} لـ {_money(hi)}">', STYLE if style else "",
             f'<path class="ar" d="{area}"/><path class="ln" d="{d}"/>']
    last = pts[-1]
    if not compact:
        for v in sorted({lo, hi}):  # recessive guides at the lowest and highest price, labelled on the side
            parts.append(f'<line class="gd" x1="{left}" x2="{left + w}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                         f'<text class="tx{" lo" if v == lo and lo != hi else ""}" x="{left + w + 4}" '
                         f'y="{y(v) + 4:.1f}">{_money(v)}</text>')
        parts.append(f'<text class="tx" x="{left}" y="{height - 6}">{_day(t0)}</text>'
                     f'<text class="tx" x="{left + w}" y="{height - 6}" text-anchor="end">النهارده</text>')
        for p in pts[1:] if older else pts:  # tooltips: a wide invisible target per point
            parts.append(f'<circle class="hit" cx="{x(p["seen_at"]):.1f}" cy="{y(p["price"]):.1f}" r="9">'
                         f'<title>{_day(p["seen_at"])}: {_money(p["price"])} ج.م</title></circle>')
    parts.append(f'<circle class="dt" cx="{x(t1):.1f}" cy="{y(last["price"]):.1f}" r="{3 if compact else 4.5}"/></svg>')
    return "".join(parts)
