import re

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def parse_price(text: str | None) -> float | None:
    """'EGP 1,299.00' / '1,299 ج.م' -> 1299.0"""
    if not text:
        return None
    text = text.translate(_AR_DIGITS)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def normalize(text: str) -> str:
    text = text.translate(_AR_DIGITS).lower()
    # "10,000 mAh" / "10000 mah" / "10k mah" -> "10000mah"
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    text = re.sub(r"(\d+)\s*k\s*mah", lambda m: f"{int(m.group(1)) * 1000}mah", text)
    text = re.sub(r"(\d+)\s*(mah|w|gb|tb|mp|hz|inch)\b", r"\1\2", text)
    return text
