

from ..driver_handler.handler import WebDriverHandler
from typing import List, Optional, Dict, TypedDict, Any, Callable, Union
from datetime import datetime
from selenium.webdriver.remote.webelement import WebElement
import re
from ..utils import get_url_info
from ..automation import (
    GoogleDriveAutomationProcessFromList
)


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


class BaseGetEcommerceProductPossiblePricesFromEcommerceProductUrl:
    @classmethod
    def get_possible_prices(
        cls,
        web_driver_handler: WebDriverHandler,
        ecommerce_product_url: EcommerceProductUrl, 
        *args,
        **kwargs,
    ) -> EcommerceProduct:
        raise NotImplementedError
    

class GenericGetEcommerceProductPossiblePricesFromEcommerceProductUrl(BaseGetEcommerceProductPossiblePricesFromEcommerceProductUrl):
    
    @staticmethod
    def get_price_restrictions(price_restrictions: PriceRestrictions=None) -> PriceRestrictions:
        if not price_restrictions:
            price_restrictions = {}
        price_restrictions = {**DefaultPriceRestrictions, **price_restrictions}
        return price_restrictions
    
    @classmethod
    def apply_price_restrictions(
        cls,
        element: WebElement,
        window_size: Dict[str, int]=None,
        price_restrictions: PriceRestrictions=None,
    ) -> Optional[PriceElement]:
        
        # Exclude striked
        text_decoration = element.value_of_css_property('text-decoration')
        if text_decoration \
        and 'line-through' in text_decoration:
            return None
        
        price_restrictions = cls.get_price_restrictions(price_restrictions)
        
        # Get restrictions
        min_text_len = price_restrictions['min_text_len']
        max_text_len = price_restrictions['max_text_len']
        currency = price_restrictions['currency']
        min_location_x = price_restrictions['min_location_x'] if isinstance(price_restrictions['min_location_x'], int) else price_restrictions['min_location_x'](window_size) 
        max_location_x = price_restrictions['max_location_x'] if isinstance(price_restrictions['max_location_x'], int) else price_restrictions['max_location_x'](window_size) 
        min_location_y = price_restrictions['min_location_y'] if isinstance(price_restrictions['min_location_y'], int) else price_restrictions['min_location_y'](window_size) 
        max_location_y = price_restrictions['max_location_y'] if isinstance(price_restrictions['max_location_y'], int) else price_restrictions['max_location_y'](window_size) 
        min_size_width = price_restrictions['min_size_width']
        max_size_width = price_restrictions['max_size_width']
        min_size_height = price_restrictions['min_size_height']
        max_size_height = price_restrictions['max_size_height']
            
        # Location restrictions
        location = getattr(element, 'location', None)
        location_x = None
        location_y = None
        if not location:
            return None
        location_x = int(location['x'])
        location_y = int(location['y'])
        
        if location_x < min_location_x \
        or max_location_x < location_x \
        or location_y < min_location_y \
        or max_location_y < location_y:
            return None
            
        # Size restrictions
        size = getattr(element, 'size', None)
        size_height = None
        size_width = None
        if not size:
            return None
        size_height = int(size['height'])
        size_width = int(size['width'])
        
        if size_width < min_size_width \
        or max_size_width < size_width \
        or size_height < min_size_height \
        or max_size_height < size_height:
            return None
        
        # Text restrictions
        text = getattr(element, 'text', None)
        if not text:
            text = element.get_attribute('innerHTML')
        if not text:
            return None
        if len(text) < min_text_len \
        or max_text_len < len(text):
            return None
        original_text_len = len(text)
        original_text = str(text)
        
        text = text.strip().replace(' ', '')
        currencies = [currency] + CURRENCY_TO_PATTERNS_MAP[currency]
        currency_match = None
        for currency in currencies:
            if currency in text:
                currency_match = currency
                break
        price_txt = re.sub(r"[^\d\.]", "", text)
        if not price_txt:
            return None
        
        # Try match price pattern
        pattern = '^([0-9]+)(\\.[0-9]+){0,1}$'
        has_price_pattern = bool(re.match(pattern, str(price_txt)))
        if not has_price_pattern:
            return None
        price_float = float(price_txt)
            
        font_size_txt = element.value_of_css_property('font-size')
        
        # Font size restrictions
        if not font_size_txt:
            return None
        
        font_size = float(re.sub(r"[^\d]", "", font_size_txt))
        font_size_unit = str(re.sub(r'[^a-zA-Z]+', "", font_size_txt))
        
        return PriceElement(
            text=original_text,
            currency=currency_match,
            price=price_float,
            font_size=font_size,
            font_size_unit=font_size_unit,
            location_x=location_x,
            location_y=location_y,
            size_height=size_height,
            size_width=size_width,
            text_len=original_text_len,
        )
    
    @classmethod
    def construct_xpath(
        cls,
        price_restrictions: PriceRestrictions=None,
        exclude_tags_and_childs: List[str]=DEFAULT_EXCLUDE_TAGS_AND_CHILDS,
    ) -> str:
        price_restrictions = cls.get_price_restrictions(price_restrictions)
        min_text_len = price_restrictions['min_text_len']
        max_text_len = price_restrictions['max_text_len']
        currency = price_restrictions['currency']
        currencies = [currency] + CURRENCY_TO_PATTERNS_MAP[currency]
        xpath = '//body//*'
        xpath += '[(' + ' or '.join([f'contains(text(), "{curr}")' for curr in currencies]) + ')'
        xpath += f' and string-length(normalize-space(text()))>={min_text_len} and string-length(normalize-space(text()))<={max_text_len}'
        if exclude_tags_and_childs:
            xpath += ' and ' + ' and '.join([f'local-name()!="{tag}" and not(ancestor::{tag})' for tag in exclude_tags_and_childs])
        xpath += ']'
        return xpath
    
    @classmethod
    def get_possible_prices(
        cls,
        web_driver_handler: WebDriverHandler,
        ecommerce_product_url: EcommerceProductUrl, 
        price_restrictions: PriceRestrictions=None,
        top_prices_num: int=99999,
        exclude_tags_and_childs: List[str]=DEFAULT_EXCLUDE_TAGS_AND_CHILDS,
    ) -> EcommerceProduct:
        web_driver_handler.go_to_page(ecommerce_product_url)
        # Get possible price elements
        xpath = cls.construct_xpath(price_restrictions, exclude_tags_and_childs)
        elements = web_driver_handler.find_elements(xpath)
        # restrict price possibilities
        records = []
        window_size = web_driver_handler.get_driver().get_window_size()
        for el in elements:
            try:
                price_restricted = cls.apply_price_restrictions(el, window_size, price_restrictions)
                if price_restricted:
                    records.append(price_restricted)
            except Exception as exc:
                # web_driver_handler.LOGGER.debug(f"Error when execute apply_price_restrictions: {exc}")
                pass
        # drop duplicated prices
        unique_elements = []
        for record in records:
            ignore = False
            for idx, unique_record in enumerate(unique_elements):
                if record['price'] == unique_record['price']:
                    ignore = True
                    if record['font_size'] > unique_record['font_size']:
                        unique_elements[idx] = record
                        break
                    
                    elif record['font_size'] == unique_record['font_size']:
                        if record['location_y'] < unique_record['location_y']:
                            unique_elements[idx] = record
                            break
            if not ignore:
                unique_elements.append(record)
        # sort prices by priority
        unique_elements = sorted(unique_elements, key=lambda item: (-item['font_size'], item['location_y']))
        unique_top_prices = unique_elements[:top_prices_num]
        return EcommerceProduct(possible_prices=unique_top_prices)


class GenericGetEcommerceProductPossiblePricesFromEcommerceProductUrlList(GoogleDriveAutomationProcessFromList):
        
    def run(
        self, 
        web_driver_handler: WebDriverHandler, 
        ecommerce_product_url: EcommerceProductUrl,
        list_item_idx: int, 
        price_restrictions: PriceRestrictions=None,
        top_prices_num: int=99999,
        exclude_tags_and_childs: List[str]=DEFAULT_EXCLUDE_TAGS_AND_CHILDS,
    ) -> EcommerceProduct:
        ecommerce_product: EcommerceProduct = EcommerceProduct()
        ecommerce_product['datetime'] = str(datetime.now())
        ecommerce_product['url_info'] = get_url_info(ecommerce_product_url)
        response = GenericGetEcommerceProductPossiblePricesFromEcommerceProductUrl.get_possible_prices(
            web_driver_handler,
            ecommerce_product_url,
            price_restrictions,
            top_prices_num,
            exclude_tags_and_childs,
        )
        ecommerce_product: EcommerceProduct = {**ecommerce_product, **response}
        return ecommerce_product