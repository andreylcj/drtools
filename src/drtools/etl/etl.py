

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
"""Logger padrão do Airflow usado nas configurações de thread pool dos extractors/loaders."""


class BaseExtractor(ThreadRequester):
    """Classe base abstrata para todos os extractors do pipeline ETL.

    Herda de :class:`~drtools.etl.request.ThreadRequester`, fornecendo
    suporte nativo a requisições HTTP com retry e paralelismo via thread pool.

    Subclasses devem implementar o método :meth:`extract`.

    Attributes:
        name (str): Nome do extractor (retorna o nome da classe).

    Example:
        >>> class MyExtractor(BaseExtractor):
        ...     HOST = "https://api.example.com"
        ...     PATHNAME = "/v1/records"
        ...     ACCEPT_ALL_PARAMS = True
        ...
        ...     def extract(self, **kwargs):
        ...         return self.send_prep_data_parse_response(**kwargs)
        ...
        ...     def parse_response(self, response, **kwargs):
        ...         return response.json()
        ...
        ...     def parse_thread_response(self, thread_response):
        ...         return [r.response for r in thread_response]
    """

    @property
    def name(self) -> str:
        """Nome do extractor (nome da classe)."""
        return self.__class__.__name__

    @abstractmethod
    def extract(self, *args, **kwargs) -> Any:
        """Executa a extração de dados. **Deve ser implementado.**

        Returns:
            Dados extraídos em qualquer formato (dict, list, DataFrame, etc.).
        """
        pass


class BaseTransformer(ABC):
    """Classe base abstrata para todos os transformers do pipeline ETL.

    Fornece integração com o sistema de logging da drtools e controle de verbosidade.

    Subclasses devem implementar o método :meth:`transform`.

    Args:
        verbosity: Se ``True``, habilita logs de progresso. Padrão: ``True``.
        LOGGER: Logger da drtools. Usa logger padrão se não informado.

    Attributes:
        name (str): Nome do transformer (retorna o nome da classe).

    Example:
        >>> class UpperCaseTransformer(BaseTransformer):
        ...     def transform(self, data, **kwargs):
        ...         return [item.upper() for item in data]
        ...
        >>> t = UpperCaseTransformer()
        >>> t.transform(["hello", "world"])
        ['HELLO', 'WORLD']
    """

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
        """Nome do transformer (nome da classe)."""
        return self.__class__.__name__

    @abstractmethod
    def transform(self, data: Any, *args, **kwargs) -> Any:
        """Transforma os dados recebidos. **Deve ser implementado.**

        Args:
            data: Dados de entrada a serem transformados.

        Returns:
            Dados transformados.
        """
        pass


class BaseLoader(ThreadRequester):
    """Classe base abstrata para todos os loaders do pipeline ETL.

    Herda de :class:`~drtools.etl.request.ThreadRequester`, fornecendo
    suporte nativo a requisições HTTP com retry e paralelismo via thread pool.

    Subclasses devem implementar o método :meth:`load`.

    Attributes:
        name (str): Nome do loader (retorna o nome da classe).

    Example:
        >>> class MyLoader(BaseLoader):
        ...     HOST = "https://api.example.com"
        ...     PATHNAME = "/v1/ingest"
        ...     ACCEPT_ALL_PARAMS = True
        ...
        ...     def load(self, data, **kwargs):
        ...         return self.send_prep_data_parse_response(data=data, **kwargs)
        ...
        ...     def parse_response(self, response, **kwargs):
        ...         return response.status_code
        ...
        ...     def parse_thread_response(self, thread_response):
        ...         return [r.response for r in thread_response]
        ...
        ...     def prep_data(self, data):
        ...         import json
        ...         return json.dumps(data)
    """

    @property
    def name(self) -> str:
        """Nome do loader (nome da classe)."""
        return self.__class__.__name__

    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        """Carrega os dados no destino. **Deve ser implementado.**

        Returns:
            Resultado da operação de carga.
        """
        pass


class BaseAPIExtractor(BaseExtractor):
    """Extractor pré-configurado para APIs HTTP.

    Implementa :meth:`extract` delegando para
    :meth:`~drtools.etl.request.BaseRequester.send_prep_data_parse_response`
    e :meth:`thread_extract` para extração paralela via thread pool.

    Subclasses ainda precisam implementar :meth:`parse_response` e,
    opcionalmente, :meth:`parse_thread_response` (se for usar thread_extract).

    Example:
        >>> class ProductsExtractor(BaseAPIExtractor):
        ...     HOST = "https://api.example.com"
        ...     PATHNAME = "/v1/products"
        ...     ACCEPT_ALL_PARAMS = True
        ...
        ...     def parse_response(self, response, **kwargs):
        ...         return response.json().get("items", [])
        ...
        ...     def parse_thread_response(self, thread_response):
        ...         result = []
        ...         for r in thread_response:
        ...             if not r.error:
        ...                 result.extend(r.response)
        ...         return result
        ...
        >>> extractor = ProductsExtractor()
        ...
        >>> # Extração simples
        >>> data = extractor.extract(
        ...     url_params=URLParams([URLParam("category", "electronics")])
        ... )
        ...
        >>> # Extração paralela (uma requisição por página)
        >>> workers = [
        ...     RequestWorker(url_params=URLParams([URLParam("page", str(p))]))
        ...     for p in range(1, 6)
        ... ]
        >>> all_data = extractor.thread_extract(request_workers=workers)
    """

    def extract(
        self,
        data: Optional[str]=None,
        url_params: URLParams=URLParams(),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        **kwargs
    ) -> Any:
        """Extrai dados de uma API via requisição HTTP simples.

        Args:
            data: Payload da requisição (string serializada).
            url_params: Parâmetros de query string.
            headers: Headers HTTP.
            http_method: Método HTTP. Padrão: GET.
            **kwargs: Argumentos adicionais repassados ao ``requests``.

        Returns:
            Resultado retornado por :meth:`parse_response`.
        """
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
        """Extrai dados de uma API em paralelo usando thread pool.

        Distribui cada :class:`~drtools.etl.request.RequestWorker` para uma thread
        independente, todas chamando :meth:`extract`.

        Args:
            request_workers: Lista de workers, cada um representando uma requisição.
            thread_config: Configuração do thread pool (max_workers, verbose, LOGGER).
                           Padrão: 12 workers com verbose a cada 100 iterações.
            headers: Headers compartilhados por todas as requisições.
            http_method: Método HTTP. Padrão: GET.
            **kwargs: Argumentos adicionais repassados a cada chamada de :meth:`extract`.

        Returns:
            Resultado de :meth:`parse_thread_response`.
        """
        return self._thread_send(
            send_method=self.extract,
            request_workers=request_workers,
            thread_config=thread_config,
            headers=headers,
            http_method=http_method,
            **kwargs
        )


class To(DefaultAssignReceivedValues):
    """Container de loaders para uso em workflows ETL.

    Recebe uma lista de classes de :class:`BaseLoader` e as registra como
    atributos da instância pelo nome da classe, facilitando o acesso
    declarativo dentro de um workflow.

    Args:
        loaders: Lista de classes de loader a registrar.

    Example:
        >>> class MyLoader(BaseLoader):
        ...     def load(self, data, **kwargs): ...
        ...     def parse_response(self, response, **kwargs): ...
        ...     def parse_thread_response(self, tr): ...
        ...     def prep_data(self, data): ...
        ...
        >>> to = To([MyLoader])
        >>> to.MyLoader  # acessa a classe MyLoader diretamente
        <class '__main__.MyLoader'>
    """

    def __init__(
        self,
        loaders: List[BaseLoader]=None
    ) -> None:
        super(To, self).__init__(loaders)


class Through(DefaultAssignReceivedValues):
    """Container de transformers para uso em workflows ETL.

    Recebe uma lista de classes de :class:`BaseTransformer` e as registra como
    atributos da instância pelo nome da classe.

    Args:
        transformers: Lista de classes de transformer a registrar.

    Example:
        >>> class CleanTransformer(BaseTransformer):
        ...     def transform(self, data, **kwargs):
        ...         return [d for d in data if d]
        ...
        >>> through = Through([CleanTransformer])
        >>> through.CleanTransformer  # acessa a classe diretamente
        <class '__main__.CleanTransformer'>
    """

    def __init__(
        self,
        transformers: List[BaseTransformer]=None
    ) -> None:
        super(Through, self).__init__(transformers)


class From(DefaultAssignReceivedValues):
    """Container de extractors para uso em workflows ETL.

    Recebe uma lista de classes de :class:`BaseExtractor` e as registra como
    atributos da instância pelo nome da classe.

    Args:
        extractors: Lista de classes de extractor a registrar.

    Example:
        >>> class MyExtractor(BaseExtractor):
        ...     def extract(self, **kwargs): ...
        ...     def parse_response(self, response, **kwargs): ...
        ...     def parse_thread_response(self, tr): ...
        ...
        >>> frm = From([MyExtractor])
        >>> frm.MyExtractor  # acessa a classe diretamente
        <class '__main__.MyExtractor'>
    """

    def __init__(
        self,
        extractors: List[BaseExtractor]=None
    ) -> None:
        super(From, self).__init__(extractors)


class BaseToFromTransformer(BaseTransformer):
    """Transformer com referências declarativas a containers de origem e destino.

    Exige que os atributos estáticos :attr:`To` e :attr:`From` sejam definidos
    na subclasse antes da instanciação, garantindo que o transformer conheça
    explicitamente de onde vêm e para onde vão os dados.

    Class Attributes:
        To (To): Container de loaders destino. **Obrigatório.**
        From (From): Container de extractors origem. **Obrigatório.**

    Raises:
        AssertionError: Se ``To`` ou ``From`` não estiverem definidos.

    Example:
        >>> class MyTransformer(BaseToFromTransformer):
        ...     To = To([MyLoader])
        ...     From = From([MyExtractor])
        ...
        ...     def transform(self, data, **kwargs):
        ...         return [item["value"] * 2 for item in data]
    """

    To: To = None
    From: From = None

    def __init__(self, *args, **kwargs) -> None:
        assert self.To is not None, \
            "Static attribute To must be set."
        assert self.From is not None, \
            "Static attribute From must be set."
        super(BaseToFromTransformer, self).__init__(*args, **kwargs)


class BaseToFromDataframeTransformer(BaseToFromTransformer):
    """Transformer ETL com pipeline JSON → DataFrame → JSON e suporte a From/To.

    Implementa :meth:`transform` como um pipeline completo:

    1. :meth:`pre_validate` — valida os dados brutos.
    2. :meth:`parsejson2dataframe` — converte JSON para :class:`pandas.DataFrame`.
    3. :meth:`transform_dataframe` — aplica a transformação no DataFrame (**abstrato**).
    4. :meth:`parsedataframe2json` — converte o DataFrame transformado de volta para JSON.

    Subclasses devem implementar :meth:`transform_dataframe` e, opcionalmente,
    :meth:`parsejson2dataframe` e :meth:`pre_validate`.

    Example:
        >>> class DoubleValueTransformer(BaseToFromDataframeTransformer):
        ...     To = To([MyLoader])
        ...     From = From([MyExtractor])
        ...
        ...     def transform_dataframe(self, dataframe):
        ...         dataframe["value"] = dataframe["value"] * 2
        ...         return dataframe
        ...
        >>> import pandas as pd
        >>> t = DoubleValueTransformer()
        >>> data = [{"value": 1}, {"value": 2}, {"value": 3}]
        >>> result = t.transform(pd.DataFrame(data))
        >>> result  # [{"value": 2}, {"value": 4}, {"value": 6}]
    """

    def verbose(self, message: str):
        """Loga ``message`` se ``verbosity=True``."""
        if self.verbosity:
            self.LOGGER.info(message)

    def pre_validate(self, data: JSONLike) -> Tuple[bool, Optional[str]]:
        """Valida os dados antes do pipeline de transformação.

        Por padrão, retorna inválido se a lista estiver vazia.

        Args:
            data: Dados de entrada no formato JSON (list of dicts).

        Returns:
            Tupla ``(is_valid, message)``. ``message`` é ``None`` quando válido.
        """
        if len(data) == 0:
            return False, "No data to transform."
        return True, None

    def parsejson2dataframe(self, data: JSONLike) -> DataFrame:
        """Converte dados JSON para :class:`pandas.DataFrame`.

        Por padrão, retorna ``data`` diretamente (assumindo que já é um DataFrame).
        Sobrescreva para customizar a conversão.

        Args:
            data: Dados no formato JSON ou já um DataFrame.

        Returns:
            :class:`pandas.DataFrame`.
        """
        return data

    @abstractmethod
    def transform_dataframe(self, dataframe: DataFrame) -> DataFrame:
        """Aplica a transformação no DataFrame. **Deve ser implementado.**

        Args:
            dataframe: DataFrame de entrada.

        Returns:
            DataFrame transformado.
        """
        pass

    def parsedataframe2json(self, dataframe: DataFrame) -> JSONLike:
        """Converte um :class:`pandas.DataFrame` para lista de dicts (JSON records).

        Usa ``orient='records'`` e ``date_format='iso'``.

        Args:
            dataframe: DataFrame a ser serializado.

        Returns:
            Lista de dicts com os registros do DataFrame.
        """
        data_list = dataframe.to_json(orient='records', date_format='iso')
        data_list = json.loads(data_list)
        return data_list

    def transform(self, data: JSONLike) -> JSONLike:
        """Executa o pipeline completo de transformação JSON → DataFrame → JSON.

        Pipeline:
            1. Loga o tamanho dos dados recebidos.
            2. Chama :meth:`pre_validate` — interrompe e retorna ``[]`` se inválido.
            3. Chama :meth:`parsejson2dataframe`.
            4. Chama :meth:`transform_dataframe`.
            5. Chama :meth:`parsedataframe2json`.

        Args:
            data: Lista de dicts (JSON records).

        Returns:
            Lista de dicts transformados, ou ``[]`` se os dados forem inválidos.
        """
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
    """Loader pré-configurado para APIs HTTP.

    Implementa :meth:`load` delegando para
    :meth:`~drtools.etl.request.BaseRequester.send_prep_data_parse_response`
    e :meth:`thread_load` para carga paralela via thread pool.

    Subclasses devem implementar :meth:`prep_data`, :meth:`parse_response` e,
    opcionalmente, :meth:`parse_thread_response` (se for usar thread_load).

    Example:
        >>> class IngestLoader(BaseAPILoader):
        ...     HOST = "https://api.example.com"
        ...     PATHNAME = "/v1/ingest"
        ...     ACCEPT_ALL_PARAMS = True
        ...
        ...     def prep_data(self, data):
        ...         import json
        ...         return json.dumps({"records": data})
        ...
        ...     def parse_response(self, response, **kwargs):
        ...         return response.status_code
        ...
        ...     def parse_thread_response(self, thread_response):
        ...         return [r.response for r in thread_response]
        ...
        >>> loader = IngestLoader()
        ...
        >>> # Carga simples
        >>> status = loader.load(data=[{"id": 1, "name": "foo"}])
        ...
        >>> # Carga paralela
        >>> workers = [
        ...     RequestWorker(data=chunk)
        ...     for chunk in chunked_data
        ... ]
        >>> statuses = loader.thread_load(request_workers=workers)
    """

    def load(
        self,
        data: Any,
        url_params: URLParams=URLParams(),
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.POST,
        **kwargs
    ) -> Any:
        """Carrega ``data`` em uma API via requisição HTTP simples.

        Args:
            data: Dados a serem carregados (qualquer tipo; será processado por :meth:`prep_data`).
            url_params: Parâmetros de query string.
            headers: Headers HTTP.
            http_method: Método HTTP. Padrão: POST.
            **kwargs: Argumentos adicionais repassados ao ``requests``.

        Returns:
            Resultado retornado por :meth:`parse_response`.
        """
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
        """Carrega dados em uma API em paralelo usando thread pool.

        Distribui cada :class:`~drtools.etl.request.RequestWorker` para uma thread
        independente, todas chamando :meth:`load`.

        Args:
            request_workers: Lista de workers, cada um representando uma carga.
            thread_config: Configuração do thread pool (max_workers, verbose, LOGGER).
                           Padrão: 12 workers com verbose a cada 100 iterações.
            headers: Headers compartilhados por todas as requisições.
            http_method: Método HTTP. Padrão: POST.
            **kwargs: Argumentos adicionais repassados a cada chamada de :meth:`load`.

        Returns:
            Resultado de :meth:`parse_thread_response`.
        """
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
        """Serializa/prepara o payload antes do envio. **Deve ser implementado.**

        Args:
            data: Dados brutos a serem preparados.

        Returns:
            Payload serializado como string (ou ``None``).
        """
        pass
