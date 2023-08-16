

from typing import List, Dict, Union, Optional, Any, Callable
from drtools.logging import Logger, FormatterOptions
from requests import Response, Session
import traceback
import json
from copy import deepcopy
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from enum import Enum
from drtools.thread_pool_executor import (
    ThreadPoolExecutor, 
    ThreadConfig, 
    WorkerResponse
)
from drtools.types import (
    JSONLike
)


class URLParam:
    def __init__(
        self,
        name: str,
        value: Optional[str]=None
    ) -> None:
        self.name = name
        self.value = value
    
    @property
    def url_param(self) -> str:
        return f'{self.name}={self.value}'


class URLParams:
    def __init__(
        self,
        url_params: List[URLParam]=[]
    ):
        self._url_params = {
            url_param.name: url_param
            for url_param in url_params
        }
        
    def replace_url_param(self, url_param: URLParam):
        if self.has_url_param(url_param):
            self._url_params[url_param.name] = url_param
    
    def add_url_param(self, url_param: URLParam):
        self._url_params[url_param.name] = url_param
    
    def remove_url_param_by_name(self, name: str):
        if name in self._url_params:
            del self._url_params[name]
        
    def get_url_param_by_name(self, name: str) -> URLParam:
        return self._url_params[name]
    
    def list_url_params(self) -> List[URLParam]:
        return [v for k, v in self._url_params.items()]
    
    def list_url_params_names(self) -> List[URLParam]:
        return [url_param.name for url_param in self.list_url_params()]
    
    def has_url_param(self, url_param: URLParam) -> bool:
        return self._url_params.get(url_param.name, None) is not None
    
    def has_url_param_by_name(self, name: str) -> bool:
        return self._url_params.get(name, None) is not None
        
    def add_if_not_exist(self, url_param: URLParam):
        if not self.has_url_param(url_param=url_param):
            self.add_url_param(url_param=url_param)
    
    def build(self) -> str:
        return '&'.join([url_param.url_param for url_param in self.list_url_params()])


class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    # DELETE = "DELETE"
    
    
class BaseRequester:
    
    HOST: str = None
    PATHNAME: str = None
    INCLUDE_IF_NOT_PROVIDED_PARAMS: URLParams = URLParams()
    ACCEPTABLE_PARAMS: URLParams = URLParams()
    ACCEPT_ALL_PARAMS: bool = False
    DATA_SAMPLE_LENGTH: int = 250
    
    def __init__(
        self,
        retry: Retry=Retry(1),
        LOGGER: Logger=Logger(
                name="Requester",
                formatter_options=FormatterOptions(
                    include_datetime=True,
                    include_thread_name=True,
                    include_logger_name=True,
                    include_level_name=True,
                ),
                default_start=False
            ),
    ) -> None:
        assert self.HOST is not None, \
            "Static attribute HOST must be set."
        assert self.PATHNAME is not None, \
            "Static attribute PATHNAME must be set."
        self.retry = retry
        self.LOGGER = LOGGER
        self.URL = None
    
    def prep_data(self, data: Any) -> Optional[str]:
        return data
    
    def _build_url(
        self, 
        url: str,
        url_params_str: Optional[str]=None
    ) -> str:
        # url = f'{self.HOST}{self.PATHNAME}'
        if not url.endswith('/'):
            url = f'{url}/'
        if url_params_str:
            url = f'{url}?{url_params_str}'
        return url
    
    def build_url(
        self, 
        data: Optional[str]=None,
        url_params: URLParams=URLParams(),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        extra: Dict={},
        **kwargs
    ) -> str:
        url_params_str = url_params.build()
        url = f'{self.HOST}{self.PATHNAME}'
        self.URL = self._build_url(
            url=url,
            url_params_str=url_params_str
        )
        return self.URL
    
    def build_headers(
        self, 
        data: Optional[str]=None,
        url_params: URLParams=URLParams(),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        extra: Dict={},
        **kwargs
    ) -> Dict:
        return headers
    
    def put_credentials_on_headers(
        self, 
        data: Optional[str]=None,
        url_params: URLParams=URLParams(),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        extra: Dict={},
        **kwargs
    ) -> Dict:
        return headers
    
    def _mount_session(
        self,
        headers: Dict={},
    ) -> Session:
        session = Session()
        adapter = HTTPAdapter(max_retries=self.retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        if headers:
            session.headers.update(headers)
        return session
    
    def _data_sample(
        self,
        data: Optional[str]=None,
    ) -> str:
        data_str = str(data)
        if len(data_str) < 2*self.DATA_SAMPLE_LENGTH:
            return f'{data_str}'
        return f'{data_str[:self.DATA_SAMPLE_LENGTH]} ... {data_str[-self.DATA_SAMPLE_LENGTH:]}'
    
    def request(
        self, 
        url: str,
        data: Optional[str]=None,
        url_params: URLParams=URLParams(),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        extra: Dict={},
        **kwargs
    ) -> Union[Response, Exception]:
        session = self._mount_session(headers)
        http_method_val = http_method.value
        response = None
        
        try:
            self.LOGGER.info(f'Executing: {http_method_val} {url}')
            if data is not None:
                self.LOGGER.info(f'Data Sample: {self._data_sample(data)}')
                kwargs['data'] = data
            if headers:
                kwargs['headers'] = headers
            request_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in [
                    'params',
                    'data',
                    'headers',
                    'cookies',
                    'files',
                    'auth',
                    'timeout',
                    'allow_redirects',
                    'proxies',
                    'hooks',
                    'stream',
                    'verify',
                    'cert',
                    'json',
                ]
            }
            response = session.request(method=http_method_val, url=url, **request_kwargs)
            self.LOGGER.info(f'HTTP Response: {response}')
            self.LOGGER.info('Executing... Done!')
        
        except Exception as exc:
            response = exc
            response_txt = str(response)
            traceback_txt = traceback.format_exc().rstrip().lstrip()
            self.LOGGER.error(f'When performing {http_method_val} {url} the following exception was generated: {response_txt}')
            self.LOGGER.error(f'{traceback_txt}')
            raise exc
            
        return response
    
    def _pre_validate(
        self, 
        data: Optional[str]=None,
        url_params: URLParams=URLParams(),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        extra: Dict={},
        **kwargs
    ):
        if not self.ACCEPT_ALL_PARAMS:
            acceptable_url_param_names = self.ACCEPTABLE_PARAMS.list_url_params_names() \
                + self.INCLUDE_IF_NOT_PROVIDED_PARAMS.list_url_params_names()
            for url_param_name in url_params.list_url_params_names():
                if url_param_name not in acceptable_url_param_names:
                    raise Exception(f"Param {url_param_name} is not acceptable")
                
        if data is not None \
        and type(data) != str:
            raise TypeError(f"Data type must be str, received {type(data)}. Data Sample: {self._data_sample(data)}")
    
    def _preprocess_url_params(
        self, 
        data: Optional[str]=None,
        url_params: URLParams=URLParams(),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        extra: Dict={},
        **kwargs
    ):
        for url_param in self.INCLUDE_IF_NOT_PROVIDED_PARAMS.list_url_params():
            url_params.add_if_not_exist(url_param=url_param)
        return url_params
    
    def decode_response_content_utf8(
        self, 
        response: Response, 
        **kwargs
    ) -> JSONLike:
        
        if not hasattr(response.content, 'decode'):
            raise Exception(f"Content Error: Response.content has not attribute called 'decode'.")
        
        else:
            decoded_content_utf8 = response.content.decode("utf-8")
        
        self.LOGGER.info(f'API response content Sample: {self._data_sample(decoded_content_utf8)}')
            
        self.LOGGER.info(f'API Status Code: {response.status_code}')        
        parsed_response = json.loads(decoded_content_utf8)
        return parsed_response
    
    def send(
        self, 
        data: Optional[str]=None,
        url_params: URLParams=URLParams(),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        extra: Dict={},
        **kwargs
    ) -> Any:
        copy_url_params: URLParams = deepcopy(url_params)
        self._pre_validate(
            data=data, 
            url_params=copy_url_params, 
            headers=headers, 
            http_method=http_method,
            extra=extra,
            **kwargs
        )
        copy_url_params = self._preprocess_url_params(
            data=data, 
            url_params=copy_url_params, 
            headers=headers, 
            http_method=http_method, 
            extra=extra,
            **kwargs
        )
        url: str = self.build_url(
            data=data, 
            url_params=copy_url_params, 
            headers=headers, 
            http_method=http_method, 
            extra=extra,
            **kwargs
        )
        headers: Dict = self.build_headers(
            data=data, 
            url_params=copy_url_params, 
            headers=headers, 
            http_method=http_method, 
            extra=extra,
            **kwargs
        )
        headers: Dict = self.put_credentials_on_headers(
            data=data, 
            url_params=copy_url_params, 
            headers=headers, 
            http_method=http_method, 
            extra=extra,
            **kwargs
        )
        response: Union[Response, Exception] = self.request(
            url=url,
            data=data, 
            url_params=copy_url_params, 
            headers=headers, 
            http_method=http_method, 
            extra=extra,
            **kwargs
        )
        return response
    
    def send_prep_data_parse_response(
        self, 
        data: Optional[str]=None,
        url_params: URLParams=URLParams(),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        extra: Dict={},
        **kwargs
    ) -> Any:
        prepared_data = self.prep_data(data=data)
        response: Union[Response, Exception] = self.send(
            data=prepared_data, 
            url_params=url_params, 
            headers=headers, 
            http_method=http_method, 
            extra=extra,
            **kwargs
        )
        parsed_response: Any = self.parse_response(
            response=response, 
            url=self.URL,
            data=prepared_data, 
            url_params=url_params, 
            headers=headers, 
            http_method=http_method, 
            extra=extra,
            **kwargs
        )
        return parsed_response
    
    def parse_response(
        self, 
        response: Response, 
        url: Optional[str]=None,
        data: Optional[str]=None,
        url_params: URLParams=URLParams(),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        extra: Dict={},
        **kwargs
    ) -> Any:
        raise NotImplementedError


class ResponseContentDecodeUtf8Requester(BaseRequester):
    def parse_response(self, response: Response, **kwargs) -> JSONLike:
        return self.decode_response_content_utf8(response=response, **kwargs)


class PrepDataJSONDumpsRequester(BaseRequester):    
    def prep_data(self, data: Any) -> Optional[str]:
        return json.dumps(data)


class RequestWorker:
    def __init__(
        self,
        data: Optional[Any]=None,
        url_params: URLParams=URLParams(), 
        extra: Dict=None     
    ) -> None:
        if extra is None:
            extra = {}
        self.data = data
        self.url_params = url_params
        self.extra = extra
    
    
class ThreadRequester(BaseRequester):
    
    def parse_thread_response(
        self, 
        thread_response: List[WorkerResponse], 
    ) -> Any:
        raise NotImplementedError
    
    def _thread_send(
        self, 
        send_method: Callable,
        request_workers: List[RequestWorker],
        thread_config: ThreadConfig=ThreadConfig(
                max_workers=12,
                verbose=100,
                LOGGER=Logger(
                    name="ThreadRequester",
                    formatter_options=FormatterOptions(
                        include_datetime=True,
                        include_thread_name=True,
                        include_logger_name=True,
                        include_level_name=True,
                    ),
                    default_start=False
                ),
            ),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        **kwargs
    ) -> List[WorkerResponse]:
        
        self.LOGGER.info('Requesting by thread...')
        def exec_func(worker: RequestWorker):
            return send_method(
                data=worker.data,
                url_params=worker.url_params,
                headers=headers,
                http_method=http_method,
                extra=worker.extra,
                **kwargs
            )
        thread_config.archive_worker_response = True
        thread_pool_executor = ThreadPoolExecutor(
            exec_func=exec_func,
            worker_data=request_workers,
            thread_config=thread_config
        )
        thread_pool_executor.start()
        thread_response: List[WorkerResponse] = thread_pool_executor.get_worker_responses()        
        self.LOGGER.info('Requesting by thread... Done!')
        
        self.LOGGER.info('Parsing thread response...')
        parsed_response = self.parse_thread_response(thread_response)
        self.LOGGER.info('Parsing thread response... Done!')
        
        return parsed_response
    
    # def thread_send(
    #     self, 
    #     request_workers: List[RequestWorker],
    #     thread_config: ThreadConfig=ThreadConfig(
    #             max_workers=12,
    #             verbose=100,
    #             LOGGER=Logger(
    #                 name="ThreadRequester",
    #                 formatter_options=FormatterOptions(
    #                     include_datetime=True,
    #                     include_thread_name=True,
    #                     include_logger_name=True,
    #                     include_level_name=True,
    #                 ),
    #                 default_start=False
    #             ),
    #         ),
    #     headers: Dict={},
    #     http_method: HTTPMethod=HTTPMethod.GET,
    #     **kwargs
    # ) -> List[WorkerResponse]:
    #     return self._thread_send(
    #         send_method=self.send,
    #         request_workers=request_workers,
    #         thread_config=thread_config,
    #         headers=headers,
    #         http_method=http_method,
    #         **kwargs
    #     )
    
    # def thread_send_prep_data_parse_response(
    #     self, 
    #     request_workers: List[RequestWorker],
    #     thread_config: ThreadConfig=ThreadConfig(
    #             max_workers=12,
    #             verbose=100,
    #             LOGGER=Logger(
    #                 name="ThreadRequester",
    #                 formatter_options=FormatterOptions(
    #                     include_datetime=True,
    #                     include_thread_name=True,
    #                     include_logger_name=True,
    #                     include_level_name=True,
    #                 ),
    #                 default_start=False
    #             ),
    #         ),
    #     headers: Dict={},
    #     http_method: HTTPMethod=HTTPMethod.GET,
    #     **kwargs
    # ) -> List[WorkerResponse]:
    #     return self._thread_send(
    #         send_method=self.send_prep_data_parse_response,
    #         request_workers=request_workers,
    #         thread_config=thread_config,
    #         headers=headers,
    #         http_method=http_method,
    #         **kwargs
    #     )