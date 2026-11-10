from .btech import BTech
from .jumia import Jumia

# Jumia answers 403 to non-browser requests, so it stays off until we use its affiliate feed.
ALL_STORES = [BTech()]
