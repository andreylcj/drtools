

from .types import UrlInfo
import tldextract
import urllib.parse as urlparse


def get_url_info(url: str) -> UrlInfo:
    """Extrai e estrutura informações de uma URL usando ``tldextract`` e ``urllib.parse``.

    Combina a extração de subdomínio/domínio/sufixo do ``tldextract`` com os
    campos de esquema, params, query e fragment do ``urllib.parse``.

    Args:
        url: URL completa a ser analisada.

    Returns:
        :class:`~drtools.web_automation.types.UrlInfo` com todos os campos preenchidos.

    Example:
        >>> info = get_url_info("https://www.amazon.com/dp/B08N5WRWNW?th=1#reviews")
        >>> info["scheme"]
        'https'
        >>> info["subdomain"]
        'www'
        >>> info["domain"]
        'amazon'
        >>> info["suffix"]
        'com'
        >>> info["query"]
        'th=1'
        >>> info["fragment"]
        'reviews'
        >>> info["is_private"]
        False

        >>> # URL com subdomínio customizado
        >>> info = get_url_info("https://api.myservice.com.br/v1/items?page=2")
        >>> info["domain"]
        'myservice'
        >>> info["suffix"]
        'com.br'
        >>> info["subdomain"]
        'api'
    """
    url_extract = tldextract.extract(url)
    url_parse = urlparse.urlparse(url)
    url_info = UrlInfo(
        url=url,
        subdomain=url_extract.subdomain,
        domain=url_extract.domain,
        suffix=url_extract.suffix,
        is_private=url_extract.is_private,
        scheme=url_parse.scheme,
        params=url_parse.params,
        query=url_parse.query,
        fragment=url_parse.fragment,
    )
    return url_info
