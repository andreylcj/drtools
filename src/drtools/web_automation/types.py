

from typing import List, Any, TypedDict, Optional, Callable, Dict, Union, Tuple
from datetime import datetime


class AutomationResult(TypedDict):
    """Resultado de uma execução única de :class:`~drtools.web_automation.automation.BaseAutomationProcess`.

    Preenchido automaticamente pelo método ``__call__`` após a execução.

    Attributes:
        id: ID único de execução (UUID).
        started_at: Timestamp de início da execução (ISO string).
        finished_at: Timestamp de fim da execução (ISO string).
        result: Valor retornado pelo método :meth:`run` da automação.
        extra: Dados extras opcionais (preenchidos em ``post_run`` se necessário).

    Example:
        >>> result: AutomationResult = {
        ...     "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        ...     "started_at": "2026-03-20 10:00:00",
        ...     "finished_at": "2026-03-20 10:00:45",
        ...     "result": {"scraped_items": 150},
        ...     "extra": None,
        ... }
    """

    id: str
    started_at: str
    finished_at: str
    result: Any
    extra: Dict


class AutomationFromListItemResult(TypedDict):
    """Resultado de uma execução individual dentro de
    :class:`~drtools.web_automation.automation.BaseAutomationProcessFromList`.

    Gerado para cada item da lista processada, registrando sucesso ou falha.

    Attributes:
        id: ID único desta execução individual (UUID).
        started_at: Timestamp de início do item.
        finished_at: Timestamp de fim do item.
        error: Mensagem de erro, ou ``None`` se bem-sucedido.
        error_traceback: Traceback completo em caso de erro, ou ``None``.
        list_item_result: Valor retornado pelo método ``run`` para este item.
        list_item: Cópia do item que foi processado.

    Example:
        >>> item_result: AutomationFromListItemResult = {
        ...     "id": "abc-123",
        ...     "started_at": "2026-03-20 10:00:00",
        ...     "finished_at": "2026-03-20 10:00:05",
        ...     "error": None,
        ...     "error_traceback": None,
        ...     "list_item_result": {"price": 29.99},
        ...     "list_item": "https://example.com/product/1",
        ... }
    """

    id: str
    started_at: str
    finished_at: str
    error: Optional[str]
    error_traceback: Optional[str]
    list_item_result: Any
    list_item: Any


class AutomationFromListResult(TypedDict):
    """Resultado consolidado de uma execução de
    :class:`~drtools.web_automation.automation.BaseAutomationProcessFromList`.

    Agrega estatísticas e os resultados individuais de cada item processado.

    Attributes:
        success_count: Número de itens processados com sucesso.
        error_count: Número de itens que falharam.
        success_rate: Taxa de sucesso de 0.0 a 1.0 (preenchida ao final).
        automation_results: Lista com o resultado de cada item processado.

    Example:
        >>> result: AutomationFromListResult = {
        ...     "success_count": 48,
        ...     "error_count": 2,
        ...     "success_rate": 0.96,
        ...     "automation_results": [...],
        ... }
    """

    success_count: int
    error_count: int
    success_rate: float
    automation_results: List[AutomationFromListItemResult]


class PriceElement(TypedDict):
    """Elemento de preço extraído de uma página de e-commerce.

    Representa um candidato a preço encontrado na página, com informações
    de texto, moeda, posição e dimensões do elemento no DOM.

    Attributes:
        text: Texto original do elemento.
        currency: Código da moeda identificada (e.g. ``"USD"``), ou ``None``.
        price: Valor numérico do preço extraído.
        font_size: Tamanho da fonte do elemento.
        font_size_unit: Unidade da fonte (e.g. ``"px"``, ``"rem"``).
        location_x: Coordenada X do elemento na viewport.
        location_y: Coordenada Y do elemento na viewport.
        size_height: Altura do elemento em pixels.
        size_width: Largura do elemento em pixels.
        text_len: Comprimento do texto original.

    Example:
        >>> price: PriceElement = {
        ...     "text": "$29.99",
        ...     "currency": "USD",
        ...     "price": 29.99,
        ...     "font_size": 24.0,
        ...     "font_size_unit": "px",
        ...     "location_x": 800,
        ...     "location_y": 350,
        ...     "size_height": 30,
        ...     "size_width": 100,
        ...     "text_len": 6,
        ... }
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
    """Filtros aplicados sobre elementos candidatos a preço durante scraping de e-commerce.

    Todos os campos de localização e dimensão aceitam inteiros fixos ou callables
    que recebem ``window_size`` (dict com ``"width"`` e ``"height"``) e retornam int.

    Attributes:
        min_text_len: Comprimento mínimo do texto.
        max_text_len: Comprimento máximo do texto.
        currency: Moeda esperada (e.g. ``"USD"``).
        min_location_x: X mínimo do elemento (absoluto ou relativo à janela).
        max_location_x: X máximo do elemento.
        min_location_y: Y mínimo do elemento.
        max_location_y: Y máximo do elemento.
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
"""Restrições de preço padrão para scraping de e-commerce em USD.

Filtros aplicados:
- Texto entre 2 e 30 caracteres.
- Moeda: USD (aceita ``"$"`` e ``"US"``).
- Posição X: a partir de 35% da largura da janela.
- Posição Y: entre 15% e 120% da altura da janela.
- Largura do elemento: 5–250px.
- Altura do elemento: 5–75px.
"""


CURRENCY_TO_PATTERNS_MAP: Dict[str, List[str]] = {
    'USD': ['$', 'US']
}
"""Mapa de moeda para padrões de texto aceitos na identificação de preços.

Cada moeda mapeia para uma lista de strings que, quando presentes no texto
do elemento, indicam que a moeda foi identificada.

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
"""Tags HTML excluídas (e seus descendentes) na busca de candidatos a preço.

Evita falsos positivos em elementos como preços riscados (``<s>``, ``<strike>``),
scripts e estilos.
"""


class EcommerceProduct(TypedDict):
    """Resultado do scraping de preços de um produto de e-commerce.

    Attributes:
        datetime: Timestamp da coleta (ISO string).
        extra: Dados extras opcionais sobre o produto.
        url_info: Informações estruturadas da URL (ver :class:`UrlInfo`).
        possible_prices: Lista de candidatos a preço encontrados na página.

    Example:
        >>> product: EcommerceProduct = {
        ...     "datetime": "2026-03-20T10:30:00",
        ...     "extra": {},
        ...     "url_info": {"domain": "amazon", "suffix": "com", ...},
        ...     "possible_prices": [
        ...         {"text": "$29.99", "price": 29.99, "currency": "USD", ...}
        ...     ],
        ... }
    """

    datetime: str
    extra: Dict[str, Any]
    url_info: str
    possible_prices: List[PriceElement]


EcommerceProductUrl = str
"""Alias para URL de produto de e-commerce.

Example:
    >>> url: EcommerceProductUrl = "https://www.amazon.com/dp/B08N5WRWNW"
"""


class Worker(TypedDict):
    """Contexto de execução de um item dentro de
    :class:`~drtools.web_automation.automation.BaseAutomationProcessFromList`.

    Agrupa todos os dados necessários para processar um item da lista de forma
    independente (suporta execução paralela via thread pool).

    Attributes:
        automation_from_list_result: Resultado acumulado da automação (compartilhado entre workers).
        list_item: O item atual sendo processado.
        list_item_idx: Índice do item na lista original.
        started_at: Timestamp de início da execução da lista completa.
        total: Total de itens na lista.
        args: Argumentos posicionais extras repassados ao método ``run``.
        kwargs: Keyword arguments extras repassados ao método ``run``.
    """

    automation_from_list_result: AutomationFromListResult
    list_item: Any
    list_item_idx: int
    started_at: datetime
    total: int
    args: Tuple
    kwargs: Dict


class UrlInfo(TypedDict):
    """Informações estruturadas extraídas de uma URL via ``tldextract`` e ``urllib.parse``.

    Retornado por :func:`~drtools.web_automation.utils.get_url_info`.

    Attributes:
        url: URL original completa.
        subdomain: Subdomínio (e.g. ``"www"``, ``"api"``).
        domain: Domínio principal (e.g. ``"amazon"``).
        suffix: TLD/sufixo (e.g. ``"com"``, ``"com.br"``).
        is_private: ``True`` se o domínio for privado/local.
        scheme: Protocolo (e.g. ``"https"``).
        params: Parâmetros de caminho da URL.
        query: Query string (e.g. ``"page=1&size=50"``).
        fragment: Fragmento/âncora da URL (e.g. ``"section-2"``).

    Example:
        >>> info: UrlInfo = {
        ...     "url": "https://www.amazon.com/dp/B08N5WRWNW?th=1",
        ...     "subdomain": "www",
        ...     "domain": "amazon",
        ...     "suffix": "com",
        ...     "is_private": False,
        ...     "scheme": "https",
        ...     "params": "",
        ...     "query": "th=1",
        ...     "fragment": "",
        ... }
    """

    url: str
    subdomain: str
    domain: str
    suffix: str
    is_private: bool
    scheme: str
    params: str
    query: str
    fragment: str
