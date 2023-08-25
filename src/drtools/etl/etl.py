

from typing import List, Dict, Optional, Any, Tuple
from pandas import DataFrame
from drtools.logging import Logger, FormatterOptions
import json
from abc import ABC, abstractmethod
import logging
from drtools.thread_pool_executor import (
    ThreadConfig, 
    WorkerResponse
)
from drtools.etl.request import (
    ThreadRequester,
    HTTPMethod,
    URLParams,
    RequestWorker,    
)
from drtools.types import (
    JSONLike
)
from drtools.etl.resources import (
    DefaultAssignReceivedValues
)


AIRFLOW_LOGGER = logging.getLogger("airflow.task")


class BaseExtractor(ThreadRequester):
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @abstractmethod
    def extract(self, *args, **kwargs) -> Any:
        pass


class BaseTransformer(ABC):
    
    def __init__(
        self,
        verbosity: bool=True,
        LOGGER: Logger=None
    ) -> None:
        self.verbosity = verbosity
        if LOGGER is None:
            self.LOGGER = Logger(
                name="BaseTransformer",
                formatter_options=FormatterOptions(
                    include_datetime=True,
                    include_thread_name=True,
                    include_logger_name=True,
                    include_level_name=True,
                ),
                default_start=False
            ),
        else:
            self.LOGGER = LOGGER
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @abstractmethod
    def transform(self, data: Any, *args, **kwargs) -> Any:
        pass


class BaseLoader(ThreadRequester):
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        pass


class BaseAPIExtractor(BaseExtractor):
    
    def extract(
        self, 
        data: Optional[str]=None,
        url_params: URLParams=URLParams(),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        **kwargs
    ) -> Any:
        return self.send_prep_data_parse_response(
            data=data, 
            url_params=url_params, 
            headers=headers, 
            http_method=http_method, 
            **kwargs
        )
    
    def thread_extract(
        self, 
        request_workers: List[RequestWorker],
        thread_config: ThreadConfig=ThreadConfig(
                max_workers=12,
                verbose=100,
                LOGGER=AIRFLOW_LOGGER
            ),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        **kwargs
    ) -> List[WorkerResponse]:
        return self._thread_send(
            send_method=self.extract,
            request_workers=request_workers,
            thread_config=thread_config,
            headers=headers,
            http_method=http_method,
            **kwargs
        )


class To(DefaultAssignReceivedValues):
    def __init__(
        self,
        loaders: List[BaseLoader]=None
    ) -> None:
        super(To, self).__init__(loaders)


class Through(DefaultAssignReceivedValues):
    def __init__(
        self,
        transformers: List[BaseTransformer]=None
    ) -> None:
        super(Through, self).__init__(transformers)


class From(DefaultAssignReceivedValues):
    def __init__(
        self,
        extractors: List[BaseExtractor]=None
    ) -> None:
        super(From, self).__init__(extractors)

    
class BaseToFromTransformer(BaseTransformer):
    
    To: To = None
    From: From = None
    
    def __init__(self, *args, **kwargs) -> None:
        assert self.To is not None, \
            "Static attribute To must be set."
        assert self.From is not None, \
            "Static attribute From must be set."
        super(BaseToFromTransformer, self).__init__(*args, **kwargs)
    
    
class BaseToFromDataframeTransformer(BaseToFromTransformer):
        
    def verbose(self, message: str):
        if self.verbosity:
            self.LOGGER.info(message)
    
    def pre_validate(self, data: JSONLike) -> Tuple[bool, Optional[str]]:
        if len(data) == 0:
            return False, "No data to transform."
        return True, None
    
    def parsejson2dataframe(self, data: JSONLike) -> DataFrame:
        return data
    
    @abstractmethod
    def transform_dataframe(self, dataframe: DataFrame) -> DataFrame:
        pass
    
    def parsedataframe2json(self, dataframe: DataFrame) -> JSONLike:
        data_list = dataframe.to_json(orient='records', date_format='iso')
        data_list = json.loads(data_list)
        return data_list
    
    def transform(self, data: JSONLike) -> JSONLike:
        self.verbose(f'Received data length: {len(data):,}')
        is_valid: bool = False
        message: Optional[str] = None
        is_valid, message = self.pre_validate(data=data)
        
        if not is_valid:
            self.verbose(f'Received data is INVALID. Reason: {message}')
            return []
        
        self.verbose(f'Received data is VALID')
        
        self.verbose(f'Parsing received JSON data to DataFrame...')
        dataframe = self.parsejson2dataframe(data=data)
        self.LOGGER.info(f'Parsed DataFrame Shape: {dataframe.shape}')
        self.verbose(f'Parsing received JSON data to DataFrame... Done!')
        
        self.verbose(f'Transforming DataFrame...')
        transformed_dataframe = self.transform_dataframe(dataframe=dataframe)
        self.LOGGER.info(f'DataFrame Shape: {transformed_dataframe.shape}')
        self.verbose(f'Transforming DataFrame...')
        
        self.verbose(f'Parsing Transformed DataFrame to JSON...')
        json_data = self.parsedataframe2json(dataframe=transformed_dataframe)
        self.LOGGER.info(f'Final data length: {len(json_data)}')
        self.verbose(f'Parsing Transformed DataFrame to JSON... Done!')
        
        return json_data


class BaseAPILoader(ABC, BaseLoader):
    
    def load(
        self, 
        data: Any,
        url_params: URLParams=URLParams(),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.POST,
        **kwargs
    ) -> Any:
        return self.send_prep_data_parse_response(
            data=data, 
            url_params=url_params, 
            headers=headers, 
            http_method=http_method, 
            **kwargs
        )
    
    def thread_load(
        self, 
        request_workers: List[RequestWorker],
        thread_config: ThreadConfig=ThreadConfig(
                max_workers=12,
                verbose=100,
                LOGGER=AIRFLOW_LOGGER
            ),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.POST,
        **kwargs
    ) -> List[WorkerResponse]:
        return self._thread_send(
            send_method=self.load,
            request_workers=request_workers,
            thread_config=thread_config,
            headers=headers,
            http_method=http_method,
            **kwargs
        )
    
    @abstractmethod
    def prep_data(self, data: Any) -> Optional[str]:
        pass