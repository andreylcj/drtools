

from typing import List, Any, TypedDict, Optional, Callable, Dict, Union, Tuple
from datetime import datetime


class AutomationResult(TypedDict):
    id: str
    started_at: str
    finished_at: str
    result: Any


class AutomationFromListItemResult(TypedDict):
    started_at: str
    finished_at: str
    error: Optional[str]
    error_traceback: Optional[str]
    list_item_result: Any
    list_item: Any


class AutomationFromListResult(TypedDict):
    success_count: int
    error_count: int
    success_rate: float
    automation_results: List[AutomationFromListItemResult]


class PriceElement(TypedDict):
    text: str
    currency: str
    price: float
    font_size: float
    font_size_unit: str
    location_x: int
    location_y: int
    size_height: int
    size_width: int
    text_len: int


class PriceRestrictions(TypedDict):
    min_text_len: int
    max_text_len: int
    currency: str
    min_location_x: Union[int, Callable]
    max_location_x: Union[int, Callable]
    min_location_y: Union[int, Callable]
    max_location_y: Union[int, Callable]
    min_size_width: int
    max_size_width: int
    min_size_height: int
    max_size_height: int


DefaultPriceRestrictions: PriceRestrictions = PriceRestrictions(
    min_text_len=2,
    max_text_len=30,
    currency='USD',
    min_location_x=lambda window_size: int(0.35*window_size['width']),
    max_location_x=99999,
    min_location_y=lambda window_size: int(0.15*window_size['height']),
    max_location_y=lambda window_size: int(1.2*window_size['height']),
    min_size_width=5,
    max_size_width=250,
    min_size_height=5,
    max_size_height=75,
)


CURRENCY_TO_PATTERNS_MAP: Dict[str, List[str]] = {
    'USD': ['$', 'US']
}


DEFAULT_EXCLUDE_TAGS_AND_CHILDS: List[str] = [
    'script', 
    'noscript', 
    'style', 
    'head', 
    'strike', 
    's'
]


class EcommerceProduct(TypedDict):
    datetime: str
    extra: Dict[str, Any]
    url_info: str
    possible_prices: List[PriceElement]


EcommerceProductUrl = str

        
class Worker(TypedDict):
    automation_from_list_result: AutomationFromListResult
    list_item: Any
    list_item_idx: int
    started_at: datetime
    total: int
    args: Tuple
    kwargs: Dict


class UrlInfo(TypedDict):
    url: str
    subdomain: str
    domain: str
    suffix: str
    is_private: bool
    scheme: str
    params: str
    query: str
    fragment: str