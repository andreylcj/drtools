# Imports
import requests
from copy import deepcopy
from typing import Dict, Optional, Tuple, List, TypedDict
from enum import Enum
import re


# Types
FinalUrl = str
PostData = Dict
# PerformGetResponse = Tuple[FinalUrl, requests.Response]
# PerformPostResponse = Tuple[FinalUrl, PostData, requests.Response]


class PerformRequestResponse(TypedDict):
    """Structured response returned by a performed HTTP request."""

    final_url: str
    post_data: Optional[Dict]
    response: requests.Response
    json_response: Optional[Dict]


class HttpMethod(Enum):
    GET = ('GET',)
    POST = ('POST',)
    
    @property
    def type_name(self) -> str:
        return self.value[0]



class ApiEndpoint:
    """Defines a single API endpoint with its method, path, and default parameters."""

    @staticmethod
    def construct_string_params_from_dict_params(
        params: Dict,
        ignore_if_param_value_is_equal_to=None
    ) -> str:
        """Convert a dict of parameters to a URL query string.

        Args:
            params: Key-value pairs to serialize.
            ignore_if_param_value_is_equal_to: Skip entries whose value equals this.

        Returns:
            Query string without leading '?', e.g. 'key1=val1&key2=val2'.
        """
        str_params = ''
        for k, v in params.items():
            if ignore_if_param_value_is_equal_to == v:
                continue
            str_params += f'{k}={v}&'
        return str_params[:-1]
    
    @classmethod
    def construct_query_params_string(
        cls,
        params: Dict=None,
        ignore_if_param_value_is_equal_to=None
    ) -> str:
        """Build a full query string from a params dict, returning '' if params is empty.

        Args:
            params: Optional dict of query parameters.
            ignore_if_param_value_is_equal_to: Skip entries whose value equals this.

        Returns:
            Query string without leading '?', or empty string if no params.
        """
        final_url = ''
        if params:
            final_url = cls.construct_string_params_from_Dict_params(
                params=params,
                ignore_if_param_value_is_equal_to=ignore_if_param_value_is_equal_to,
            )
        return final_url
    
    def __init__(
        self,
        name: str,
        path: str,
        method: HttpMethod,
        default_params: Optional[Dict]=None,
        default_post_data: Optional[Dict]=None,
    ):
        """Initialize an API endpoint.

        Args:
            name: Unique identifier for this endpoint. Must match [A-Za-z0-9\\-]+.
            path: URL path (e.g. '/api/v1/resource').
            method: HTTP method (HttpMethod.GET or HttpMethod.POST).
            default_params: Default query parameters merged with per-call params.
            default_post_data: Default POST body merged with per-call post data.

        Raises:
            Exception: If name contains characters outside [A-Za-z0-9\\-].
        """
        if default_params is None:
            default_params = {}
        self.name = name
        self.path = path
        self.method = method
        self.default_params = default_params
        self.default_post_data = default_post_data
        self._validate()
    
    @staticmethod
    def validate_name(name: str) -> bool:
        """Return True if name matches the allowed pattern [A-Za-z0-9\\-]+."""
        return bool(re.fullmatch(r'[A-Za-z0-9\-]+', name))
    
    def _validate(self):
        if not self.validate_name(self.name):
            raise Exception(f"Parameter 'name' must have only a-z, A-z, 0-9, '-'. Received: {self.name}")
    
    def _contruct_endpoint_with_params(self, params: Dict=None) -> str:
        final_params = deepcopy(self.default_params)
        if params:
            final_params = {**final_params, **params}
        query_params_string = self.construct_query_params_string(params=final_params)
        final_url = self.path
        if query_params_string:
            final_url = f'{self.path}?{query_params_string}'
        return final_url
    
    def construct_post_data(
        self,
        post_data: Dict=None,
        # ignore_none: bool=True,
    ) -> Dict:
        """Merge given post_data on top of default_post_data, removing None values.

        Args:
            post_data: Per-call POST body fields.

        Returns:
            Merged dict with None values removed, or default_post_data if no overrides.
        """
        if not post_data:
            return self.default_post_data
        final_post_data = deepcopy(self.default_post_data)
        if not final_post_data:
            final_post_data = {}
        final_post_data = {**final_post_data, **post_data}
        final_post_data_cp = deepcopy(final_post_data)
        for k, v in final_post_data.items():
            if v is None:
                del final_post_data_cp[k]
        return final_post_data_cp
    
    def endpoint_url(self, params: Dict=None) -> str:
        """Return the full path with query string built from default + given params.

        Args:
            params: Optional per-call params merged on top of default_params.

        Returns:
            Path with query string, e.g. '/api/v1/resource?foo=bar'.
        """
        return self._contruct_endpoint_with_params(params=params)


class RequestApiHandler:
    """Base handler for making HTTP requests against a REST API.

    Dynamically exposes registered endpoints as callable attributes.
    """

    @staticmethod
    def perform_get(url: str) -> requests.Response:
        """Execute an HTTP GET request.

        Args:
            url: Full URL to request.

        Returns:
            The requests.Response object.
        """
        return requests.get(url)

    @staticmethod
    def perform_post(url: str, post_data: Dict=None) -> requests.Response:
        """Execute an HTTP POST request with a JSON body.

        Args:
            url: Full URL to request.
            post_data: Dict to send as JSON body.

        Returns:
            The requests.Response object.
        """
        return requests.post(url, json=post_data)

    def __init__(
        self,
        root_api_path: str,
        return_only_json_response: bool=True,
        # ignore_none_when_constructing_post_data: bool=True,
    ):
        """Initialize the handler.

        Args:
            root_api_path: Base URL prepended to all endpoint paths (e.g. 'http://localhost:8080').
            return_only_json_response: If True, calls return the parsed JSON dict directly
                instead of the full PerformRequestResponse.
            ignore_none_when_constructing_post_data: Passed to construct_post_data to strip None values.
        """
        self.root_api_path = root_api_path
        self.return_only_json_response = return_only_json_response
        # self.ignore_none_when_constructing_post_data = ignore_none_when_constructing_post_data
        self._api_endpoints = {}
        self._api_code_names = {}
        
    def add_api_endpoint(self, api_endpoint: ApiEndpoint) -> bool:
        """Register an endpoint and expose it as a method on this handler.

        The endpoint name (with '-' replaced by '_') becomes a callable attribute,
        e.g. an endpoint named 'get-status' becomes self.get_status(...).

        Args:
            api_endpoint: The endpoint to register.

        Raises:
            Exception: If an endpoint with the same name is already registered.
        """
        if api_endpoint.name in self._api_endpoints:
            raise Exception(f'Api Endpoints must have unique names. Name {api_endpoint.name} already exists.')
        self._api_endpoints[api_endpoint.name] = api_endpoint
        code_name = api_endpoint.name.replace('-', '_')
        self._api_code_names[code_name] = api_endpoint
        setattr(
            self, 
            code_name, 
            self.wrap_request(api_endpoint.name)
        )
    
    def _construct_final_url(self, endpoint_path: str) -> str:
        return self.root_api_path + endpoint_path
    
    def _perform_request(self, api_endpoint_name: str, params: Dict=None, post_data: Dict=None) -> PerformRequestResponse:
        api_endpoint = self._api_endpoints[api_endpoint_name]
        method = api_endpoint.method
        final_url = self._construct_final_url(api_endpoint.endpoint_url(params))
        final_post_data = None
        req_response = None
        if method is HttpMethod.GET:
            req_response = self.perform_get(final_url)
        elif method is HttpMethod.POST:
            final_post_data = api_endpoint.construct_post_data(
                post_data, 
                # self.ignore_none_when_constructing_post_data
            )
            req_response = self.perform_post(final_url, final_post_data)
        else:
            raise Exception(f'Method [{method}] is not allowed.')
        response = PerformRequestResponse(
            final_url=final_url,
            post_data=final_post_data,
            response=req_response,
            json_response=None,
        )
        if self.return_only_json_response:
            response = response['response'].json()
        return response
    
    def wrap_request(self, name: str) -> str:
        """Return a callable that invokes _perform_request for the given endpoint name.

        Args:
            name: Registered endpoint name.

        Returns:
            Callable(*args, **kwargs) that forwards to _perform_request.
        """
        def _request(*args, **kwargs):
            return self._perform_request(name, *args, **kwargs)
        return _request

    def list_api_methods(self) -> List[str]:
        """Return the list of registered endpoint code names (snake_case)."""
        return [x for x in self._api_code_names.keys()]