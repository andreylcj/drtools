

from perfmkt.resources.request.main import ApiEndpoint, HttpMethod


STATUS_ENDPOINT = ApiEndpoint(
    name='status', 
    path='/status', 
    method=HttpMethod.GET
)


QUERY_GROUP_ENDPOINT = ApiEndpoint(
    name='query-group', 
    path='/api/v1/group/list', 
    method=HttpMethod.GET,
    default_params={
        'group_name': None, 
        'page': None, 
        'page_size': 10
    }
)


QUERY_PROFILE_V2_ENDPOINT = ApiEndpoint(
    name='query-profile-v2', 
    path='/api/v2/browser-profile/list', 
    method=HttpMethod.POST,
    default_post_data={
        'group_id': None,
        'profile_id': None,
        'profile_no': None,
        'sort_type': None,
        'sort_order': None,
        'page': None,
        'limit': 10
    }
)


QUERY_PROXY_ENDPOINT = ApiEndpoint(
    name='query-proxy', 
    path='/api/v2/proxy-list/list', 
    method=HttpMethod.POST,
    default_post_data={
        'Proxy_id': None,
        'limit': 50,
        'page': None,
    }
)


NEW_PROFILE_V2_ENDPOINT = ApiEndpoint(
    name='new-profile-v2', 
    path='/api/v2/browser-profile/create', 
    method=HttpMethod.POST,
    default_post_data={
        'name': None,
        'group_id': None,
        'remark': None,
        'platform': None,
        'username': None,
        'password': None,
        'fakey': None,
        'cookie': None,
        'repeat_config': None,
        'ignore_cookie_error': None,
        'tabs': None,
        'user_proxy_config': None,
        'proxyid': None,
        'ip': None,
        'country': None,
        'region': None,
        'city': None,
        'ipchecker': None,
        'sys_app_cate_id': None,
        'fingerprint_config': None,
    }
)


OPEN_BROWSER_V2_ENDPOINT = ApiEndpoint(
    name='open-browser-v2', 
    path='/api/v2/browser-profile/start', 
    method=HttpMethod.POST,
    default_post_data={
        'profile_id': None,
        'profile_no': None,
        'launch_args': None,
        'headless': None,
        'last_opened_tabs': None,
        'proxy_detection': None,
        'password_filling': None,
        'password_saving': None,
        'cdp_mask': None,
        'delete_cache': None,
        'device_scale': None,
    }
)


CLOSE_BROWSER_V2_ENDPOINT = ApiEndpoint(
    name='close-browser-v2', 
    path='/api/v2/browser-profile/stop', 
    method=HttpMethod.POST,
    default_post_data={
        'profile_id': None, 
        'profile_no': None
    }
)

CHECK_BROWSER_STATUS_ENDPOINT = ApiEndpoint(
    name='check-browser-status', 
    path='/api/v1/browser/active', 
    method=HttpMethod.GET,
    default_params={
        'user_id': None, 
        'serial_number': None,
    }
)


CHECK_BROWSER_STATUS_V2_ENDPOINT = ApiEndpoint(
    name='check-browser-status-v2', 
    path='/api/v2/browser-profile/active', 
    method=HttpMethod.GET,
    default_params={
        'profile_id': None, 
        'profile_no': None,
    }
)

# Endpoints List
ENDPOINTS = [
    STATUS_ENDPOINT,
    QUERY_GROUP_ENDPOINT,
    QUERY_PROFILE_V2_ENDPOINT,
    NEW_PROFILE_V2_ENDPOINT,
    OPEN_BROWSER_V2_ENDPOINT,
    QUERY_PROXY_ENDPOINT,
    CLOSE_BROWSER_V2_ENDPOINT,
    CHECK_BROWSER_STATUS_ENDPOINT,
    CHECK_BROWSER_STATUS_V2_ENDPOINT,
]