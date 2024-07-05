

from .types import UrlInfo
import tldextract
import urllib.parse as urlparse


def get_url_info(url: str) -> UrlInfo:
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