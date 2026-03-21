

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
    """Elemento de preço extraído de uma página de e-commerce.

    Representa um candidato a preço encontrado no DOM após aplicação das
    restrições de :class:`PriceRestrictions`.

    Attributes:
        text: Texto original do elemento.
        currency: Código/símbolo da moeda identificada (e.g. ``"USD"``), ou ``None``.
        price: Valor numérico do preço extraído.
        font_size: Tamanho da fonte do elemento em CSS.
        font_size_unit: Unidade CSS da fonte (e.g. ``"px"``, ``"rem"``).
        location_x: Coordenada X do elemento na viewport.
        location_y: Coordenada Y do elemento na viewport.
        size_height: Altura do elemento em pixels.
        size_width: Largura do elemento em pixels.
        text_len: Comprimento do texto original.
    """

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
    """Filtros aplicados durante a extração de candidatos a preço.

    Campos de posição e dimensão aceitam valores inteiros fixos ou callables
    que recebem ``window_size`` (dict ``{"width": int, "height": int}``) e
    retornam int — útil para restrições relativas ao tamanho da janela.

    Attributes:
        min_text_len: Comprimento mínimo do texto.
        max_text_len: Comprimento máximo do texto.
        currency: Moeda esperada (e.g. ``"USD"``).
        min_location_x: X mínimo na viewport.
        max_location_x: X máximo na viewport.
        min_location_y: Y mínimo na viewport.
        max_location_y: Y máximo na viewport.
        min_size_width: Largura mínima do elemento.
        max_size_width: Largura máxima do elemento.
        min_size_height: Altura mínima do elemento.
        max_size_height: Altura máxima do elemento.
    """

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
"""Restrições padrão para scraping de preços em USD.

- Texto: 2–30 caracteres.
- Moeda: USD (aceita ``"$"`` e ``"US"``).
- Posição X: ≥ 35% da largura da janela.
- Posição Y: 15%–120% da altura da janela.
- Largura: 5–250px | Altura: 5–75px.
"""


CURRENCY_TO_PATTERNS_MAP: Dict[str, List[str]] = {
    'USD': ['$', 'US']
}
"""Mapa de moeda para padrões de texto que identificam a moeda no elemento.

Example:
    >>> CURRENCY_TO_PATTERNS_MAP['USD']
    ['$', 'US']
"""


DEFAULT_EXCLUDE_TAGS_AND_CHILDS: List[str] = [
    'script',
    'noscript',
    'style',
    'head',
    'strike',
    's'
]
"""Tags HTML excluídas (e todos os seus descendentes) na busca XPath de candidatos a preço.

Evita falsos positivos em elementos como preços riscados (``<s>``, ``<strike>``),
scripts e estilos inline.
"""


class EcommerceProduct(TypedDict):
    """Resultado do scraping de preços de um produto de e-commerce.

    Attributes:
        datetime: Timestamp ISO da coleta.
        extra: Dados extras opcionais sobre o produto.
        url_info: Informações estruturadas da URL (ver :func:`~drtools.web_automation.utils.get_url_info`).
        possible_prices: Candidatos a preço encontrados, ordenados por tamanho de fonte
            descendente e posição Y ascendente.
    """

    datetime: str
    extra: Dict[str, Any]
    url_info: str
    possible_prices: List[PriceElement]


EcommerceProductUrl = str
"""Alias para URL de página de produto de e-commerce."""


class BaseGetEcommerceProductPossiblePricesFromEcommerceProductUrl:
    """Interface base para extratores de preços de e-commerce.

    Define o contrato para classes que implementam a extração de preços
    a partir de uma URL de produto.

    Subclasses devem implementar :meth:`get_possible_prices`.
    """

    @classmethod
    def get_possible_prices(
        cls,
        web_driver_handler: WebDriverHandler,
        ecommerce_product_url: EcommerceProductUrl,
        *args,
        **kwargs,
    ) -> EcommerceProduct:
        """Extrai preços de uma página de produto. **Deve ser implementado.**

        Args:
            web_driver_handler: Handler com o driver ativo.
            ecommerce_product_url: URL da página do produto.

        Raises:
            NotImplementedError: Sempre que não for sobrescrito.
        """
        raise NotImplementedError


class GenericGetEcommerceProductPossiblePricesFromEcommerceProductUrl(BaseGetEcommerceProductPossiblePricesFromEcommerceProductUrl):
    """Extrator genérico de preços de e-commerce baseado em heurísticas de DOM.

    Usa XPath dinâmico para encontrar todos os elementos de texto que contêm
    padrões de moeda e aplica uma série de filtros (posição, tamanho, texto,
    CSS, padrão numérico) para identificar os melhores candidatos a preço.

    Algoritmo:
        1. Constrói XPath filtrando por padrões de moeda e comprimento de texto.
        2. Busca todos os elementos no DOM.
        3. Aplica :meth:`apply_price_restrictions` em cada elemento.
        4. Remove duplicatas — quando dois elementos têm o mesmo preço, mantém
           o de maior fonte; se igual, o de menor Y (mais alto na página).
        5. Ordena por ``(-font_size, location_y)`` — maior fonte primeiro.
        6. Retorna os ``top_prices_num`` primeiros.

    Example:
        >>> from drtools.web_automation.driver_handler.chrome import ChromeWebDriverHandler
        >>> from drtools.web_automation.gallery.ecommerce_product import (
        ...     GenericGetEcommerceProductPossiblePricesFromEcommerceProductUrl
        ... )
        ...
        >>> handler = ChromeWebDriverHandler()
        >>> handler.start(remove_ui=True, load_images=False)
        ...
        >>> product = GenericGetEcommerceProductPossiblePricesFromEcommerceProductUrl.get_possible_prices(
        ...     web_driver_handler=handler,
        ...     ecommerce_product_url="https://www.amazon.com/dp/B08N5WRWNW",
        ...     top_prices_num=3,
        ... )
        ...
        >>> for price in product["possible_prices"]:
        ...     print(price["text"], price["price"], price["font_size"])
        ...
        >>> handler.quit()
    """

    @staticmethod
    def get_price_restrictions(price_restrictions: PriceRestrictions=None) -> PriceRestrictions:
        """Mescla restrições customizadas com os defaults.

        Args:
            price_restrictions: Restrições customizadas (podem ser parciais).
                Se ``None`` ou vazio, usa :data:`DefaultPriceRestrictions` integralmente.

        Returns:
            :class:`PriceRestrictions` completo com os defaults preenchidos para
            campos não especificados.

        Example:
            >>> restrictions = GenericGetEcommerceProductPossiblePricesFromEcommerceProductUrl.get_price_restrictions(
            ...     {"max_text_len": 15, "currency": "USD"}
            ... )
        """
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
        """Avalia se um elemento DOM é um candidato válido a preço.

        Aplica filtros em sequência — qualquer falha retorna ``None``:

        1. **CSS text-decoration**: Exclui elementos com ``line-through`` (preços riscados).
        2. **Localização XY**: Verifica se o elemento está dentro dos limites da janela.
        3. **Dimensões**: Verifica largura e altura do elemento.
        4. **Texto**: Comprimento, presença de padrão de moeda e padrão numérico.
        5. **Font-size**: Presença de tamanho de fonte válido.

        Args:
            element: Elemento DOM candidato.
            window_size: Dict ``{"width": int, "height": int}`` da janela do browser.
                Necessário quando as restrições usam callables relativos à janela.
            price_restrictions: Restrições a aplicar. Se ``None``, usa os defaults.

        Returns:
            :class:`PriceElement` se o elemento passar em todos os filtros,
            ``None`` caso contrário.

        Example:
            >>> elements = handler.find_elements('//body//*[contains(text(), "$")]')
            >>> window_size = handler.get_driver().get_window_size()
            >>> prices = []
            >>> for el in elements:
            ...     result = GenericGetEcommerceProductPossiblePricesFromEcommerceProductUrl.apply_price_restrictions(
            ...         el, window_size
            ...     )
            ...     if result:
            ...         prices.append(result)
        """
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
        """Constrói o XPath para busca de candidatos a preço no DOM.

        O XPath busca todos os elementos em ``//body`` que:
        - Contêm o símbolo da moeda ou seus padrões alternativos.
        - Têm comprimento de texto dentro dos limites de :class:`PriceRestrictions`.
        - Não são (nem descendentes de) tags excluídas.

        Args:
            price_restrictions: Restrições a aplicar. Se ``None``, usa os defaults.
            exclude_tags_and_childs: Tags HTML a excluir da busca.

        Returns:
            String XPath pronta para uso em :meth:`~drtools.web_automation.driver_handler.handler.WebDriverHandler.find_elements`.

        Example:
            >>> xpath = GenericGetEcommerceProductPossiblePricesFromEcommerceProductUrl.construct_xpath()
            >>> # Resultado similar a:
            >>> # '//body//*[(contains(text(), "$") or contains(text(), "US"))
            >>> #   and string-length(normalize-space(text()))>=2
            >>> #   and string-length(normalize-space(text()))<=30
            >>> #   and local-name()!="script" and not(ancestor::script) ...]'
        """
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
        """Extrai os preços mais prováveis de uma página de produto de e-commerce.

        Pipeline completo:
        1. Navega para a URL.
        2. Constrói XPath via :meth:`construct_xpath`.
        3. Busca todos os elementos candidatos.
        4. Aplica :meth:`apply_price_restrictions` em cada elemento.
        5. Remove duplicatas (mantém maior fonte por preço único).
        6. Ordena por ``(-font_size, location_y)``.
        7. Retorna os ``top_prices_num`` primeiros.

        Args:
            web_driver_handler: Handler com driver ativo.
            ecommerce_product_url: URL da página do produto.
            price_restrictions: Restrições customizadas (parciais ou completas).
                Se ``None``, usa :data:`DefaultPriceRestrictions`.
            top_prices_num: Número máximo de preços a retornar. Padrão: ``99999`` (todos).
            exclude_tags_and_childs: Tags a excluir da busca XPath.

        Returns:
            :class:`EcommerceProduct` com ``possible_prices`` ordenados e deduplicados.

        Example:
            >>> handler = ChromeWebDriverHandler()
            >>> handler.start(remove_ui=True, load_images=False)
            ...
            >>> product = GenericGetEcommerceProductPossiblePricesFromEcommerceProductUrl.get_possible_prices(
            ...     web_driver_handler=handler,
            ...     ecommerce_product_url="https://www.amazon.com/dp/B08N5WRWNW",
            ...     top_prices_num=1,  # apenas o preço principal
            ... )
            ...
            >>> main_price = product["possible_prices"][0]
            >>> print(f"Preço: {main_price['text']} ({main_price['price']})")
            Preço: $29.99 (29.99)
            ...
            >>> handler.quit()
        """
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
    """Automação de lista que extrai preços de múltiplas URLs de produto em paralelo.

    Combina :class:`GenericGetEcommerceProductPossiblePricesFromEcommerceProductUrl`
    com :class:`~drtools.web_automation.automation.GoogleDriveAutomationProcessFromList`
    para processar uma lista de URLs, salvando os resultados automaticamente
    no Google Drive.

    Herda todo o comportamento de paralelismo, retry e upload do Drive da classe pai.

    Example:
        >>> from drtools.web_automation.gallery.ecommerce_product import (
        ...     GenericGetEcommerceProductPossiblePricesFromEcommerceProductUrlList
        ... )
        ...
        >>> class AmazonPriceScraper(GenericGetEcommerceProductPossiblePricesFromEcommerceProductUrlList):
        ...     NAME = "amazon-price-scraper"
        ...     GOOGLE_DRIVE_BASE_FOLDER_PATH = "ScraperResults"
        ...
        >>> scraper = AmazonPriceScraper(
        ...     max_workers=2,
        ...     worker_max_tries=3,
        ...     wait_time=2,
        ... )
        >>> scraper.gdrive.set_credentials("service_account.json")
        >>> scraper.start(remove_ui=True, load_images=False)
        ...
        >>> urls = [
        ...     "https://www.amazon.com/dp/B08N5WRWNW",
        ...     "https://www.amazon.com/dp/B09G9HD6PD",
        ... ]
        >>> scraper(urls, top_prices_num=3)
        >>> result = scraper.get_result()
        >>> for item in result["result"]["automation_results"]:
        ...     if not item["error"]:
        ...         print(item["list_item"], item["list_item_result"]["possible_prices"])
    """

    def run(
        self,
        web_driver_handler: WebDriverHandler,
        ecommerce_product_url: EcommerceProductUrl,
        list_item_idx: int,
        price_restrictions: PriceRestrictions=None,
        top_prices_num: int=99999,
        exclude_tags_and_childs: List[str]=DEFAULT_EXCLUDE_TAGS_AND_CHILDS,
    ) -> EcommerceProduct:
        """Processa uma URL de produto e retorna o resultado com metadados.

        Args:
            web_driver_handler: Handler com driver ativo para este worker.
            ecommerce_product_url: URL do produto a ser scrapeado.
            list_item_idx: Índice do item na lista (usado pelo framework).
            price_restrictions: Restrições customizadas de preço.
            top_prices_num: Número máximo de preços a retornar por produto.
            exclude_tags_and_childs: Tags HTML a excluir da busca.

        Returns:
            :class:`EcommerceProduct` com ``datetime``, ``url_info`` e
            ``possible_prices`` preenchidos.
        """
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
