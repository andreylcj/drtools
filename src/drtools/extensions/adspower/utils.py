

# Imports
from drtools.etl.other_request import RequestApiHandler
from typing import Dict, List
from fake_useragent import UserAgent


def find_first_proxy_with_no_profile_related(
    adspower_handler: RequestApiHandler,
    proxy_tag_name_is_some_of: List[str]=None,
    limit: int=200,
) -> Dict:
    """Return the first proxy that has no browser profile associated with it.

    Args:
        adspower_handler: An initialized RequestApiHandler with Adspower endpoints.
        proxy_tag_name_is_some_of: If provided, only proxies whose tags include one of
            these names are considered.
        limit: Number of proxies to fetch per page.

    Returns:
        The proxy dict, or None if all proxies have associated profiles.

    Raises:
        Exception: If the API response has no 'data' key or no proxies are found.
    """
    for r in range(1000):
        response = adspower_handler.query_proxy(post_data={'limit': limit, 'page': r+1})
        if 'data' not in response:
            raise Exception(f'Response has no "data". Response: {response}')
        if len(response['data']['list']) == 0:
            raise Exception('No proxy was found.')
        for p in response['data']['list']:
            if int(p['profile_count']) > 0:
                continue
            if proxy_tag_name_is_some_of:
                for t in p['proxy_tags']:
                    for tag_name in proxy_tag_name_is_some_of:
                        if tag_name == t['name']:
                            return p
            else:
                return p

        
def get_ua(
    browser_custom_version: str=None,
    browsers: List[str]=['Chrome'],
    platforms: str='desktop',
    min_version: int=120,
    os=None,
) -> str:
    """Generate a random Chrome user agent string.

    Args:
        browser_custom_version: If provided, replaces the browser version in the generated UA string.
        browsers: List of browser names to filter (default: ['Chrome']).
        platforms: Platform type to filter (default: 'desktop').
        min_version: Minimum browser major version to use.
        os: Operating system to filter (None means any).

    Returns:
        A User-Agent string, e.g. 'Mozilla/5.0 ... Chrome/120.0.0.0 Safari/537.36'.
    """
    ua = UserAgent(browsers=browsers, platforms=platforms, min_version=min_version, os=os).getRandom
    ua_string = ua['useragent']
    if browser_custom_version:
        ua_string = ua_string.replace(ua['browser_version'], browser_custom_version)
    return ua_string