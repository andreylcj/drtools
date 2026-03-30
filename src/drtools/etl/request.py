

from typing import List, Dict, Union, Optional, Any, Callable, TypedDict
from drtools.logging import Logger, FormatterOptions
from requests import Response, Session
import traceback
import json
import re
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
    """Representa um único parâmetro de query string de URL.

    Args:
        name: Nome do parâmetro (e.g. ``"page"``).
        value: Valor do parâmetro (e.g. ``"1"``). Pode ser ``None``.

    Attributes:
        url_param (str): String formatada ``"name=value"``.

    Example:
        >>> param = URLParam(name="page", value="1")
        >>> param.url_param
        'page=1'
    """

    def __init__(
        self,
        name: str,
        value: Optional[str]=None
    ) -> None:
        self.name = name
        self.value = value

    @property
    def url_param(self) -> str:
        """Retorna a representação ``"name=value"`` do parâmetro."""
        return f'{self.name}={self.value}'


class URLParams:
    """Coleção gerenciada de :class:`URLParam` usada para construir query strings.

    Mantém internamente um dicionário indexado pelo nome do parâmetro,
    garantindo unicidade por nome.

    Args:
        url_params: Lista inicial de :class:`URLParam`. Padrão: lista vazia.

    Example:
        >>> params = URLParams([URLParam("page", "1"), URLParam("size", "50")])
        >>> params.build()
        'page=1&size=50'

        >>> params.add_url_param(URLParam("filter", "active"))
        >>> params.has_url_param_by_name("filter")
        True

        >>> params.remove_url_param_by_name("size")
        >>> params.build()
        'page=1&filter=active'
    """

    def __init__(
        self,
        url_params: List[URLParam]=[]
    ):
        self._url_params = {
            url_param.name: url_param
            for url_param in url_params
        }

    def replace_url_param(self, url_param: URLParam):
        """Substitui um parâmetro existente pelo mesmo nome. Não-op se não existir."""
        if self.has_url_param(url_param):
            self._url_params[url_param.name] = url_param

    def add_url_param(self, url_param: URLParam):
        """Adiciona ou sobrescreve um parâmetro pelo nome."""
        self._url_params[url_param.name] = url_param

    def remove_url_param_by_name(self, name: str):
        """Remove o parâmetro com o nome informado, se existir."""
        if name in self._url_params:
            del self._url_params[name]

    def get_url_param_by_name(self, name: str) -> URLParam:
        """Retorna o :class:`URLParam` pelo nome. Lança ``KeyError`` se não existir."""
        return self._url_params[name]

    def list_url_params(self) -> List[URLParam]:
        """Retorna lista com todos os :class:`URLParam` registrados."""
        return [v for k, v in self._url_params.items()]

    def list_url_params_names(self) -> List[str]:
        """Retorna lista com os nomes de todos os parâmetros registrados."""
        return [url_param.name for url_param in self.list_url_params()]

    def has_url_param(self, url_param: URLParam) -> bool:
        """Verifica se já existe um parâmetro com o mesmo nome que ``url_param``."""
        return self._url_params.get(url_param.name, None) is not None

    def has_url_param_by_name(self, name: str) -> bool:
        """Verifica se existe um parâmetro com o nome informado."""
        return self._url_params.get(name, None) is not None

    def add_if_not_exist(self, url_param: URLParam):
        """Adiciona o parâmetro apenas se ainda não existir um com o mesmo nome."""
        if not self.has_url_param(url_param=url_param):
            self.add_url_param(url_param=url_param)

    def build(self) -> str:
        """Constrói e retorna a query string completa (sem ``?`` inicial).

        Example:
            >>> URLParams([URLParam("a", "1"), URLParam("b", "2")]).build()
            'a=1&b=2'
        """
        return '&'.join([url_param.url_param for url_param in self.list_url_params()])


class HTTPMethod(Enum):
    """Enum dos métodos HTTP suportados pelo módulo.

    Members:
        GET: Método HTTP GET.
        POST: Método HTTP POST.
        PUT: Método HTTP PUT.
        PATCH: Método HTTP PATCH.

    Example:
        >>> HTTPMethod.GET.value
        'GET'
    """

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"


class BaseRequester:
    """Classe base para realização de requisições HTTP com retry automático.

    Define o ciclo completo de uma requisição:
    ``prep_data`` → ``build_url`` → ``build_headers`` → ``put_credentials_on_headers``
    → ``request`` → ``parse_response``.

    Subclasses devem sobrescrever :meth:`parse_response` (obrigatório) e,
    opcionalmente, :meth:`prep_data`, :meth:`build_url`, :meth:`build_headers`
    e :meth:`put_credentials_on_headers`.

    Class Attributes:
        HOST (str): Host base da API (e.g. ``"https://api.example.com"``). Obrigatório ao usar :meth:`build_url`.
        PATHNAME (str): Caminho do endpoint (e.g. ``"/v1/users"``). Obrigatório ao usar :meth:`build_url`.
        INCLUDE_IF_NOT_PROVIDED_PARAMS (URLParams): Parâmetros adicionados automaticamente
            se não fornecidos pelo chamador.
        ACCEPTABLE_PARAMS (URLParams): Conjunto de parâmetros permitidos. Validado quando
            ``ACCEPT_ALL_PARAMS=False``.
        ACCEPT_ALL_PARAMS (bool): Se ``True``, desabilita a validação de parâmetros.
        DATA_SAMPLE_LENGTH (int): Tamanho do trecho de dados exibido nos logs. Padrão: 250.

    Args:
        retry: Configuração de retry do ``urllib3``. Padrão: 1 tentativa.
        LOGGER: Logger da drtools. Usa logger padrão se não informado.

    Example:
        >>> class MyAPI(BaseRequester):
        ...     HOST = "https://api.example.com"
        ...     PATHNAME = "/v1/items"
        ...     ACCEPT_ALL_PARAMS = True
        ...
        ...     def parse_response(self, response, **kwargs):
        ...         return response.json()
        ...
        >>> api = MyAPI()
        >>> result = api.send_prep_data_parse_response(
        ...     url_params=URLParams([URLParam("page", "1")])
        ... )
    """

    HOST: str = None
    PATHNAME: str = None
    INCLUDE_IF_NOT_PROVIDED_PARAMS: URLParams = URLParams()
    ACCEPTABLE_PARAMS: URLParams = URLParams()
    ACCEPT_ALL_PARAMS: bool = False
    DATA_SAMPLE_LENGTH: int = 250

    def __init__(
        self,
        retry: Retry=Retry(1),
        LOGGER: Logger=None,
    ) -> None:
        if not LOGGER:
            LOGGER = Logger(
                name="Requester",
                formatter_options=FormatterOptions(
                    include_datetime=True,
                    include_thread_name=True,
                    include_logger_name=True,
                    include_level_name=True,
                ),
                default_start=False
            )
        self.retry = retry
        self.LOGGER = LOGGER
        self.URL = None

    def prep_data(self, data: Any) -> Optional[str]:
        """Prepara o payload antes do envio. Por padrão retorna ``data`` sem modificação.

        Sobrescreva para serializar, criptografar ou transformar o payload.

        Args:
            data: Dados brutos a serem preparados.

        Returns:
            Dados prontos para envio (deve ser ``str`` ou ``None``).
        """
        return data

    def _build_url(
        self,
        url: str,
        url_params_str: Optional[str]=None
    ) -> str:
        """Garante trailing slash e anexa a query string à URL base.

        Args:
            url: URL base sem query string.
            url_params_str: Query string já serializada (sem ``?``).

        Returns:
            URL completa com query string (se houver).
        """
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
        """Constrói a URL final combinando ``HOST``, ``PATHNAME`` e ``url_params``.

        Armazena o resultado em ``self.URL`` e o retorna.

        Raises:
            Exception: Se ``HOST`` ou ``PATHNAME`` não estiverem definidos.

        Returns:
            URL completa pronta para uso na requisição.
        """
        url_params_str = url_params.build()
        if self.HOST is None:
            raise Exception("Static attribute HOST must be set.")
        if self.PATHNAME is None:
            raise Exception("Static attribute PATHNAME must be set.")
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
        """Constrói os headers da requisição. Por padrão retorna ``headers`` sem modificação.

        Sobrescreva para adicionar headers padrão (e.g. ``Content-Type``, ``Accept``).

        Returns:
            Dicionário de headers.
        """
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
        """Injeta credenciais nos headers. Por padrão retorna ``headers`` sem modificação.

        Sobrescreva para adicionar tokens de autenticação (Bearer, API key, etc.).

        Returns:
            Dicionário de headers com credenciais.
        """
        return headers

    def _mount_session(
        self,
        headers: Dict={},
    ) -> Session:
        """Cria e configura uma :class:`requests.Session` com retry e headers.

        Args:
            headers: Headers a serem aplicados na sessão.

        Returns:
            Sessão configurada com adaptadores HTTP/HTTPS e retry.
        """
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
        """Retorna um trecho representativo de ``data`` para uso em logs.

        Se o dado for menor que ``2 * DATA_SAMPLE_LENGTH``, retorna completo.
        Caso contrário, retorna início e fim separados por ``" ... "``.

        Args:
            data: Dado a ser amostrado.

        Returns:
            String truncada para fins de log.
        """
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
        """Executa a requisição HTTP propriamente dita via ``requests.Session``.

        Monta a sessão, loga a operação, executa o método HTTP e retorna a resposta.
        Qualquer kwargs reconhecido pelo ``requests`` (``timeout``, ``verify``,
        ``json``, ``auth``, etc.) é repassado automaticamente.

        Args:
            url: URL completa a ser requisitada.
            data: Payload serializado (string).
            url_params: Parâmetros de URL (usados apenas para contexto, URL já construída).
            headers: Headers da requisição.
            http_method: Método HTTP a ser usado.
            extra: Dados extras para uso em subclasses.
            **kwargs: Argumentos adicionais aceitos pelo ``requests`` (``timeout``,
                ``verify``, ``json``, ``auth``, ``stream``, etc.).

        Returns:
            Objeto :class:`requests.Response` em caso de sucesso.

        Raises:
            Exception: Re-levanta qualquer exceção gerada durante a requisição.
        """
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
        """Valida os parâmetros antes do envio.

        Verifica se os parâmetros fornecidos em ``url_params`` estão na lista
        de parâmetros aceitáveis (quando ``ACCEPT_ALL_PARAMS=False``).
        Também garante que ``data`` seja do tipo ``str`` se fornecido.

        Raises:
            Exception: Se um parâmetro não aceitável for encontrado.
            TypeError: Se ``data`` não for ``str``.
        """
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
        """Injeta os parâmetros padrão definidos em ``INCLUDE_IF_NOT_PROVIDED_PARAMS``.

        Parâmetros já presentes em ``url_params`` não são sobrescritos.

        Returns:
            O mesmo objeto ``url_params`` mutado com os defaults adicionados.
        """
        for url_param in self.INCLUDE_IF_NOT_PROVIDED_PARAMS.list_url_params():
            url_params.add_if_not_exist(url_param=url_param)
        return url_params

    def decode_response_content_utf8(
        self,
        response: Response,
        **kwargs
    ) -> JSONLike:
        """Decodifica o conteúdo da resposta HTTP como UTF-8 e faz parse do JSON.

        Args:
            response: Objeto :class:`requests.Response`.

        Returns:
            Objeto Python resultante do parse do JSON (dict, list, etc.).

        Raises:
            Exception: Se ``response.content`` não possuir o método ``decode``.
        """
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
        """Executa o pipeline completo de envio: validação → pré-processamento
        → construção de URL → construção de headers → credenciais → requisição.

        Faz uma cópia profunda de ``url_params`` para não modificar o original.

        Args:
            data: Payload já serializado como string.
            url_params: Parâmetros de query string.
            headers: Headers HTTP.
            http_method: Método HTTP.
            extra: Dados extras repassados para os hooks do pipeline.
            **kwargs: Argumentos adicionais repassados ao ``requests``.

        Returns:
            Objeto :class:`requests.Response`.
        """
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
        """Atalho de alto nível: prepara dados → envia → parseia resposta.

        Equivale a chamar :meth:`prep_data` + :meth:`send` + :meth:`parse_response`
        em sequência.

        Returns:
            Resultado retornado por :meth:`parse_response`.

        Example:
            >>> class MyAPI(BaseRequester):
            ...     HOST = "https://api.example.com"
            ...     PATHNAME = "/v1/items"
            ...     ACCEPT_ALL_PARAMS = True
            ...     def parse_response(self, response, **kwargs):
            ...         return response.json()
            ...
            >>> result = MyAPI().send_prep_data_parse_response()
        """
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
        """Parseia a resposta HTTP. **Deve ser implementado pelas subclasses.**

        Args:
            response: Objeto :class:`requests.Response`.
            url: URL que gerou a resposta.
            data: Payload enviado.
            url_params: Parâmetros de URL utilizados.
            headers: Headers utilizados.
            http_method: Método HTTP utilizado.
            extra: Dados extras.

        Raises:
            NotImplementedError: Sempre que não for sobrescrito.
        """
        raise NotImplementedError


class ResponseContentDecodeUtf8Requester(BaseRequester):
    """Implementação de :class:`BaseRequester` que parseia a resposta como JSON UTF-8.

    ``parse_response`` decodifica ``response.content`` em UTF-8 e faz ``json.loads``.

    Example:
        >>> class MyAPI(ResponseContentDecodeUtf8Requester):
        ...     HOST = "https://api.example.com"
        ...     PATHNAME = "/v1/data"
        ...     ACCEPT_ALL_PARAMS = True
        ...
        >>> data = MyAPI().send_prep_data_parse_response()
        >>> isinstance(data, (dict, list))
        True
    """

    def parse_response(self, response: Response, **kwargs) -> JSONLike:
        return self.decode_response_content_utf8(response=response, **kwargs)


class PrepDataJSONDumpsRequester(BaseRequester):
    """Implementação de :class:`BaseRequester` que serializa o payload com ``json.dumps``.

    ``prep_data`` converte automaticamente dicts/lists para string JSON antes do envio.

    Example:
        >>> class MyAPI(PrepDataJSONDumpsRequester):
        ...     HOST = "https://api.example.com"
        ...     PATHNAME = "/v1/ingest"
        ...     ACCEPT_ALL_PARAMS = True
        ...     def parse_response(self, response, **kwargs):
        ...         return response.status_code
        ...
        >>> api = MyAPI()
        >>> api.prep_data({"key": "value"})
        '{"key": "value"}'
    """

    def prep_data(self, data: Any) -> Optional[str]:
        return json.dumps(data)


class RequestWorker:
    """Encapsula os argumentos de uma única requisição para uso em execução paralela.

    Usado em conjunto com :class:`ThreadRequester` para submeter múltiplas
    requisições simultaneamente via thread pool.

    Args:
        data: Payload da requisição.
        url_params: Parâmetros de URL específicos desta requisição.
        extra: Dados extras repassados ao pipeline de envio.

    Example:
        >>> workers = [
        ...     RequestWorker(url_params=URLParams([URLParam("id", str(i))]))
        ...     for i in range(1, 6)
        ... ]
        >>> # Passar para thread_extract / thread_load de um extractor/loader
    """

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
    """Extensão de :class:`BaseRequester` com suporte a requisições paralelas via thread pool.

    Adiciona o método interno :meth:`_thread_send`, que distribui uma lista de
    :class:`RequestWorker` entre múltiplas threads e consolida as respostas.

    Subclasses devem implementar :meth:`parse_thread_response` para transformar
    a lista de :class:`WorkerResponse` no formato desejado.

    Example:
        >>> class MyThreadAPI(ThreadRequester):
        ...     HOST = "https://api.example.com"
        ...     PATHNAME = "/v1/items"
        ...     ACCEPT_ALL_PARAMS = True
        ...
        ...     def parse_response(self, response, **kwargs):
        ...         return response.json()
        ...
        ...     def parse_thread_response(self, thread_response):
        ...         return [r.response for r in thread_response if not r.error]
        ...
        >>> api = MyThreadAPI()
        >>> workers = [
        ...     RequestWorker(url_params=URLParams([URLParam("id", str(i))]))
        ...     for i in range(10)
        ... ]
        >>> results = api._thread_send(
        ...     send_method=api.send_prep_data_parse_response,
        ...     request_workers=workers,
        ... )
    """

    def parse_thread_response(
        self,
        thread_response: List[WorkerResponse],
    ) -> Any:
        """Processa a lista de respostas coletadas pelas threads. **Deve ser implementado.**

        Args:
            thread_response: Lista de :class:`WorkerResponse` retornada pelo thread pool.

        Raises:
            NotImplementedError: Sempre que não for sobrescrito.
        """
        raise NotImplementedError

    def _thread_send(
        self,
        send_method: Callable,
        request_workers: List[RequestWorker],
        thread_config: ThreadConfig=None,
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        **kwargs
    ) -> List[WorkerResponse]:
        """Distribui ``request_workers`` em um thread pool e coleta as respostas.

        Internamente cria um :class:`ThreadPoolExecutor`, executa ``send_method``
        para cada worker em paralelo e chama :meth:`parse_thread_response` sobre
        o resultado consolidado.

        Args:
            send_method: Método de envio a ser chamado para cada worker
                (tipicamente :meth:`send_prep_data_parse_response` ou :meth:`~BaseLoader.load`).
            request_workers: Lista de :class:`RequestWorker` com os dados de cada requisição.
            thread_config: Configuração do thread pool (max_workers, verbose, LOGGER).
            headers: Headers compartilhados por todas as requisições.
            http_method: Método HTTP compartilhado por todas as requisições.
            **kwargs: Argumentos extras repassados ao ``send_method``.

        Returns:
            Resultado de :meth:`parse_thread_response`.
        """
        if not thread_config:
            thread_config = ThreadConfig(
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
            )
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
        self.thread_response: List[WorkerResponse] = thread_pool_executor.get_worker_responses()
        self.LOGGER.info('Requesting by thread... Done!')

        self.LOGGER.info('Parsing thread response...')
        self.parsed_response = self.parse_thread_response(self.thread_response)
        self.LOGGER.info('Parsing thread response... Done!')

        return self.parsed_response


class PerformRequestResponse(TypedDict):
    """Resposta estruturada retornada por :class:`ApiRequester` quando
    ``return_only_json_response=False``.

    Attributes:
        final_url: URL completa que foi requisitada (com query string).
        post_data: Payload enviado na requisição (``None`` para GET).
        response: Objeto bruto :class:`requests.Response`.
        json_response: Resultado de ``response.json()``, ou ``None`` se
            o parse não for aplicável.

    Example:
        >>> result: PerformRequestResponse = {
        ...     "final_url": "https://api.example.com/v1/items/?page=1",
        ...     "post_data": None,
        ...     "response": <Response [200]>,
        ...     "json_response": {"items": [...]},
        ... }
    """

    final_url: str
    post_data: Optional[Dict]
    response: Response
    json_response: Optional[Dict]


class ApiEndpoint:
    """Define um endpoint nomeado de uma API REST com método, path e defaults por chamada.

    Permite registrar múltiplos endpoints em uma mesma instância de
    :class:`ApiRequester`, cada um com seus próprios ``default_params``
    (:class:`URLParams`) e ``default_post_data`` (dict).

    Args:
        name: Identificador único do endpoint. Deve corresponder ao padrão
            ``[A-Za-z0-9\\-]+``. Ao registrar em :class:`ApiRequester`, hifens
            são convertidos para underscores no nome do método gerado.
        path: Caminho do endpoint relativo ao ``HOST``
            (e.g. ``"/api/v1/users"``).
        method: Método HTTP do endpoint (:class:`HTTPMethod`).
        default_params: :class:`URLParams` com parâmetros de query padrão.
            São mesclados com os params fornecidos em cada chamada — os da
            chamada têm prioridade (sobrescrevem defaults de mesmo nome).
        default_post_data: Dict com campos padrão do corpo da requisição POST.
            Mesclado com o ``post_data`` de cada chamada; valores ``None``
            são removidos do resultado final.

    Raises:
        Exception: Se ``name`` contiver caracteres fora de ``[A-Za-z0-9\\-]``.

    Example:
        >>> endpoint = ApiEndpoint(
        ...     name="list-items",
        ...     path="/api/v1/items",
        ...     method=HTTPMethod.GET,
        ...     default_params=URLParams([URLParam("size", "50")]),
        ... )
        >>> endpoint.endpoint_url(URLParams([URLParam("page", "2")]))
        '/api/v1/items?size=50&page=2'

        >>> post_ep = ApiEndpoint(
        ...     name="create-item",
        ...     path="/api/v1/items",
        ...     method=HTTPMethod.POST,
        ...     default_post_data={"active": True, "source": "api"},
        ... )
        >>> post_ep.construct_post_data({"name": "foo"})
        {"active": True, "source": "api", "name": "foo"}
    """

    @staticmethod
    def validate_name(name: str) -> bool:
        """Verifica se ``name`` corresponde ao padrão ``[A-Za-z0-9\\-]+``.

        Returns:
            ``True`` se válido, ``False`` caso contrário.
        """
        return bool(re.fullmatch(r'[A-Za-z0-9\-]+', name))

    def __init__(
        self,
        name: str,
        path: str,
        method: HTTPMethod,
        default_params: URLParams=None,
        default_post_data: Optional[Dict]=None,
    ):
        if default_params is None:
            default_params = URLParams()
        self.name = name
        self.path = path
        self.method = method
        self.default_params = default_params
        self.default_post_data = default_post_data
        if not self.validate_name(self.name):
            raise Exception(
                f"Parameter 'name' must match [A-Za-z0-9\\-]+. Received: {self.name!r}"
            )

    def _construct_endpoint_with_params(self, params: URLParams=None) -> str:
        """Constrói o path completo com query string mesclando defaults e params da chamada.

        Params da chamada sobrescrevem defaults de mesmo nome.

        Args:
            params: :class:`URLParams` adicionais para esta chamada.

        Returns:
            Path com query string (e.g. ``"/api/v1/items?size=50&page=2"``),
            ou apenas o path se não houver params.
        """
        if params is None:
            params = URLParams()
        merged = deepcopy(self.default_params)
        for p in params.list_url_params():
            merged.add_url_param(p)
        qs = merged.build()
        return f'{self.path}?{qs}' if qs else self.path

    def endpoint_url(self, params: URLParams=None) -> str:
        """Retorna o path com query string para uso na construção da URL final.

        Args:
            params: :class:`URLParams` adicionais para esta chamada.
                Mesclados sobre os ``default_params`` do endpoint.

        Returns:
            Path relativo com query string (sem host).

        Example:
            >>> ep = ApiEndpoint("search", "/v1/search", HTTPMethod.GET,
            ...                  URLParams([URLParam("lang", "pt")]))
            >>> ep.endpoint_url(URLParams([URLParam("q", "python")]))
            '/v1/search?lang=pt&q=python'
        """
        return self._construct_endpoint_with_params(params=params)

    def construct_post_data(self, post_data: Dict=None) -> Optional[Dict]:
        """Mescla ``post_data`` da chamada sobre ``default_post_data``, removendo ``None``.

        Args:
            post_data: Campos a adicionar/sobrescrever no corpo da requisição.
                Se ``None``, retorna ``default_post_data`` diretamente.

        Returns:
            Dict mesclado sem valores ``None``, ou ``None`` se não houver dados.

        Example:
            >>> ep = ApiEndpoint("create", "/v1/items", HTTPMethod.POST,
            ...                  default_post_data={"active": True, "source": "api"})
            >>> ep.construct_post_data({"name": "foo", "active": None})
            {"source": "api", "name": "foo"}
        """
        if not post_data:
            return self.default_post_data
        final = deepcopy(self.default_post_data) if self.default_post_data else {}
        final = {**final, **post_data}
        return {k: v for k, v in final.items() if v is not None}


class ApiRequester(ThreadRequester):
    """Handler de API REST com múltiplos endpoints registrados dinamicamente.

    Combina o melhor de dois mundos:

    - **Multi-endpoint por instância** — registre N endpoints com
      :meth:`add_endpoint` e acesse cada um como método callable
      (``self.list_items()``, ``self.create_item(post_data={...})``).
    - **Infraestrutura de :class:`ThreadRequester`** — Session com retry,
      logging drtools, paralelismo via thread pool, pipeline completo
      de ``prep_data`` → ``build_url`` → ``request`` → ``parse_response``.

    Class Attributes:
        HOST: URL base da API (e.g. ``"https://api.example.com"``). **Obrigatório.**
        ACCEPT_ALL_PARAMS: Sempre ``True`` — validação de params é delegada
            a cada :class:`ApiEndpoint`.

    Args:
        return_only_json_response: Se ``True`` (padrão), os métodos de endpoint
            retornam apenas ``response.json()``. Se ``False``, retornam um
            :class:`PerformRequestResponse` completo.
        **kwargs: Repassados a :class:`ThreadRequester` (``retry``, ``LOGGER``).

    Example:
        >>> from drtools.etl.request import (
        ...     ApiRequester, ApiEndpoint, URLParams, URLParam, HTTPMethod
        ... )
        ...
        >>> class MyAPI(ApiRequester):
        ...     HOST = "https://jsonplaceholder.typicode.com"
        ...
        >>> api = MyAPI()
        ...
        >>> api.add_endpoint(ApiEndpoint(
        ...     name="list-posts",
        ...     path="/posts",
        ...     method=HTTPMethod.GET,
        ...     default_params=URLParams([URLParam("_limit", "10")]),
        ... ))
        ...
        >>> api.add_endpoint(ApiEndpoint(
        ...     name="create-post",
        ...     path="/posts",
        ...     method=HTTPMethod.POST,
        ...     default_post_data={"userId": 1},
        ... ))
        ...
        >>> # Chama o endpoint como método (nome com hífen vira snake_case)
        >>> posts = api.list_posts()
        >>> posts = api.list_posts(params=URLParams([URLParam("_limit", "5")]))
        ...
        >>> new_post = api.create_post(post_data={"title": "Hello", "body": "World"})
        ...
        >>> # Listar endpoints registrados
        >>> api.list_api_methods()
        ['list_posts', 'create_post']
        ...
        >>> # Com resposta estruturada completa
        >>> api2 = MyAPI(return_only_json_response=False)
        >>> api2.add_endpoint(ApiEndpoint("list-posts", "/posts", HTTPMethod.GET))
        >>> result = api2.list_posts()
        >>> result["final_url"]
        'https://jsonplaceholder.typicode.com/posts'
        >>> result["response"].status_code
        200
    """

    ACCEPT_ALL_PARAMS: bool = True

    def __init__(
        self,
        return_only_json_response: bool=True,
        **kwargs,
    ) -> None:
        super(ApiRequester, self).__init__(**kwargs)
        self.return_only_json_response = return_only_json_response
        self._endpoints: Dict[str, ApiEndpoint] = {}

    def add_endpoint(self, endpoint: ApiEndpoint) -> None:
        """Registra um :class:`ApiEndpoint` e expõe-o como método na instância.

        O nome do endpoint (com hifens substituídos por underscores) torna-se
        um método callable: ``add_endpoint(ApiEndpoint("get-user", ...))``
        cria ``self.get_user(params=..., post_data=...)``.

        Args:
            endpoint: Endpoint a registrar.

        Raises:
            Exception: Se já existir um endpoint com o mesmo nome.

        Example:
            >>> api.add_endpoint(ApiEndpoint(
            ...     name="get-user",
            ...     path="/v1/users/{id}",
            ...     method=HTTPMethod.GET,
            ... ))
            >>> api.get_user()  # chama o endpoint
        """
        if endpoint.name in self._endpoints:
            raise Exception(
                f"Endpoint '{endpoint.name}' already registered. Names must be unique."
            )
        self._endpoints[endpoint.name] = endpoint
        code_name = endpoint.name.replace('-', '_')
        setattr(self, code_name, self._wrap(endpoint.name))

    def _wrap(self, endpoint_name: str) -> Callable:
        """Retorna um callable que invoca :meth:`_call_endpoint` para ``endpoint_name``.

        Args:
            endpoint_name: Nome do endpoint registrado.

        Returns:
            Callable com assinatura ``(params=URLParams(), post_data=None)``.
        """
        def _call(
            params: URLParams=None,
            post_data: Dict=None,
        ) -> Any:
            return self._call_endpoint(endpoint_name, params, post_data)
        return _call

    def _call_endpoint(
        self,
        endpoint_name: str,
        params: URLParams=None,
        post_data: Dict=None,
    ) -> Any:
        """Executa uma requisição para o endpoint registrado com o nome dado.

        Mescla os params e post_data da chamada com os defaults do endpoint,
        serializa o payload e delega ao pipeline de :meth:`send_prep_data_parse_response`.

        Args:
            endpoint_name: Nome do endpoint a chamar.
            params: :class:`URLParams` adicionais para esta chamada.
            post_data: Campos do corpo POST para esta chamada.

        Returns:
            ``response.json()`` se ``return_only_json_response=True``,
            ou :class:`PerformRequestResponse` completo caso contrário.
        """
        if params is None:
            params = URLParams()
        endpoint = self._endpoints[endpoint_name]
        final_post_data = endpoint.construct_post_data(post_data)
        return self.send_prep_data_parse_response(
            data=final_post_data,
            url_params=params,
            http_method=endpoint.method,
            extra={'endpoint': endpoint},
        )

    def prep_data(self, data: Any) -> Optional[str]:
        """Serializa o payload para JSON string antes do envio.

        Retorna ``None`` para requisições sem corpo (GET).
        Serializa dicts/lists com ``json.dumps`` para POST/PUT/PATCH.

        Args:
            data: Dict ou lista a serializar, ou ``None``.

        Returns:
            String JSON, ou ``None`` se ``data`` for ``None``.
        """
        if data is None:
            return None
        return json.dumps(data)

    def build_url(
        self,
        data: Optional[str]=None,
        url_params: URLParams=None,
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        extra: Dict={},
        **kwargs,
    ) -> str:
        """Constrói a URL final usando o path do :class:`ApiEndpoint` em ``extra``.

        Combina ``HOST`` com o path+query string do endpoint. Quando ``extra``
        não contém um endpoint (uso direto de ``send``), delega para
        :meth:`BaseRequester.build_url` — exigindo ``PATHNAME`` definido.

        Args:
            url_params: :class:`URLParams` da chamada, mesclados com os defaults
                do endpoint dentro de :meth:`ApiEndpoint.endpoint_url`.
            extra: Deve conter ``{"endpoint": ApiEndpoint(...)}`` quando chamado
                via :meth:`_call_endpoint`.

        Returns:
            URL completa pronta para requisição.

        Raises:
            Exception: Se ``HOST`` não estiver definido.
        """
        if url_params is None:
            url_params = URLParams()
        endpoint: ApiEndpoint = extra.get('endpoint')
        if endpoint is None:
            return super().build_url(data, url_params, headers, http_method, extra, **kwargs)
        if self.HOST is None:
            raise Exception("Static attribute HOST must be set.")
        self.URL = self.HOST + endpoint.endpoint_url(url_params)
        return self.URL

    def parse_response(
        self,
        response: Response,
        url: Optional[str]=None,
        data: Optional[str]=None,
        url_params: URLParams=None,
        headers: Dict={},
        http_method: HTTPMethod=HTTPMethod.GET,
        extra: Dict={},
        **kwargs,
    ) -> Any:
        """Parseia a resposta HTTP conforme ``return_only_json_response``.

        Args:
            response: Objeto :class:`requests.Response`.
            url: URL final que gerou a resposta.
            data: Payload serializado enviado na requisição.
            url_params: Params utilizados.
            headers: Headers utilizados.
            http_method: Método HTTP utilizado.
            extra: Dados extras (contém o endpoint ativo).

        Returns:
            ``response.json()`` se ``return_only_json_response=True`` (padrão).
            :class:`PerformRequestResponse` completo se ``False``.
        """
        if self.return_only_json_response:
            return response.json()
        return PerformRequestResponse(
            final_url=url or self.URL,
            post_data=json.loads(data) if data else None,
            response=response,
            json_response=response.json(),
        )

    def parse_thread_response(self, thread_response) -> Any:
        """Retorna as respostas das threads como lista. Sobrescreva para customizar.

        Por padrão retorna a lista completa de :class:`WorkerResponse` sem filtragem.

        Args:
            thread_response: Lista de respostas coletadas pelo thread pool.

        Returns:
            A mesma lista ``thread_response`` recebida.
        """
        return thread_response

    def list_api_methods(self) -> List[str]:
        """Retorna os nomes em snake_case de todos os endpoints registrados.

        Returns:
            Lista de strings com os nomes dos métodos gerados dinamicamente.

        Example:
            >>> api.list_api_methods()
            ['list_posts', 'create_post', 'get_user']
        """
        return [name.replace('-', '_') for name in self._endpoints]
