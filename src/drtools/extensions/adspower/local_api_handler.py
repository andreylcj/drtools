

from . import local_api_endpoints
from drtools.etl.other_request import ApiEndpoint, RequestApiHandler
from .settings import DEFAULT_ROOT_API_PATH
from typing import Dict, List
import time
from drtools.logging import Logger, FormatterOptions


# Set endpoints
ENDPOINTS = []
for k, v in local_api_endpoints.__dict__.items():
    if isinstance(v, ApiEndpoint):
        ENDPOINTS.append(v)


class AdspowerHandler(RequestApiHandler):
    """HTTP handler pre-configured with all Adspower LocalAPI endpoints.

    All endpoints defined in local_api_endpoints are registered automatically
    and exposed as snake_case methods (e.g. self.open_browser_v2(...)).
    """

    def __init__(
        self,
        root_api_path: str=None,
        return_only_json_response: bool=True,
        LOGGER: Logger=None,
    ):
        """Initialize the Adspower handler.

        Args:
            root_api_path: Override the default Adspower API root URL.
                Defaults to DEFAULT_ROOT_API_PATH from settings.
            return_only_json_response: If True, calls return parsed JSON directly.
        """
        if root_api_path is None:
            root_api_path = DEFAULT_ROOT_API_PATH
        super(AdspowerHandler, self).__init__(
            root_api_path=root_api_path,
            return_only_json_response=return_only_json_response,
        )
        for endpoint in ENDPOINTS:
            self.add_api_endpoint(endpoint)
        if LOGGER is None:
            LOGGER = Logger(
                name='AdspowerHandler',
                formatter_options=FormatterOptions(
                    include_datetime=True,
                    include_thread_name=True,
                    include_logger_name=True,
                    include_level_name=True,
                ),
                default_start=False
            )
        self.LOGGER = LOGGER
            
    def query_profiles_v2_by_name(
        self,
        name: str,
        case: str="contains", # contains, startswith, exact,
        limit: int=100,
    ):
        reponse = []
        for r in range(1000):
            response = self.query_profile_v2(post_data={'limit': limit, 'page': r+1})
            if 'data' not in response:
                raise Exception(f'Response has no "data". Response: {response}')
            if len(response['data']['list']) == 0:
                break
                # raise Exception('No profile was found.')
            for prof in response['data']['list']:
                found_profile = None
                if case == 'contains':
                    if name in prof['name']:
                        found_profile = prof
                elif case == 'startswith':
                    if prof['name'].startswith(name):
                        found_profile = prof
                elif case == 'exact':
                    if prof['name'] == name:
                        found_profile = prof
                else:
                    raise Exception(f'Case "{case}" is not valid.')
                if found_profile:
                    reponse.append(found_profile)
            time.sleep(2)
        return reponse
    
    def get_profile_v2_by_name(
        self,
        name: str,
        case: str="contains", # contains, startswith, exact,
    ):
        profiles = self.query_profiles_v2_by_name(name, case)
        single_prof = None
        if profiles:
            if len(profiles) > 1:
                profile_ids = ', '.join([p['profile_id'] for p in profiles])
                raise Exception(f'Was found more than 1 profile with name name {name} and case {case}. Found: {len(profiles)}. Ids: {profile_ids}')
            single_prof = profiles[0]
        else:
            raise Exception(f'No profile was found with name {name} and case {case}.')
        return single_prof
            
    def open_browser_v2_by_name(
        self,
        name: str,
        case: str="contains", # contains, startswith, exact
    ):
        profile = self.get_profile_v2_by_name(name, case)
        profile_id = profile['profile_id']
        self.LOGGER.debug(f'Opening browser {profile_id}...')
        response = self.open_browser_v2(post_data={'profile_id': profile_id})
        self.LOGGER.debug(f'Opening browser {profile_id}... Done!')
        # if profiles:
        #     if len(profiles) > 1:
        #         profile_ids = ', '.join([p['profile_id'] for p in profiles])
        #         raise Exception(f'Was found more than 1 profile with name name {name} and case {case}. Found: {len(profiles)}. Ids: {profile_ids}')
        #     profile_id = profiles[0]['profile_id']
        #     self.LOGGER.debug(f'Opening browser {profile_id}...')
        #     self.browser = self.open_browser_v2(post_data={'profile_id': profile_id})
        #     self.LOGGER.debug(f'Opening browser {profile_id}... Done!')
        # else:
        #     raise Exception(f'No profile was found with name {name} and case {case}.')
        return response
    
    def find_first_proxy_with_no_profile_related(
        self,
        proxy_tag_name_is_some_of: List[str]=None,
        limit: int=200,
    ) -> Dict:
        """Return the first proxy that has no browser profile associated with it.

        Args:
            proxy_tag_name_is_some_of: If provided, only proxies whose tags include one of
                these names are considered.
            limit: Number of proxies to fetch per page.

        Returns:
            The proxy dict, or None if all proxies have associated profiles.

        Raises:
            Exception: If the API response has no 'data' key or no proxies are found.
        """
        for r in range(1000):
            response = self.query_proxy(post_data={'limit': limit, 'page': r+1})
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
            time.sleep(2)