

from .driver_handler.handler import WebDriverHandler
from .driver_handler.chrome import ChromeWebDriverHandler
from typing import List, Any, Union, Tuple
from drtools.logging import Logger, FormatterOptions
from drtools.google.drive.drive import DriveFromServiceAcountFile
import uuid
from datetime import datetime
import time
import os
from drtools.utils import display_time, retry, remove_break_line
import traceback
from selenium.webdriver.remote.webdriver import WebDriver
import random
from threading import Lock
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed as futures_as_completed
)
from .types import (
    AutomationResult,
    AutomationFromListItemResult,
    AutomationFromListResult,
    Worker,
)
from .bot_detection import BotDetection
from copy import deepcopy
from .driver_handler.config import (
    DEFAULT_BOT_DETECTION_METHODS,
    DEFAULT_BOT_DETECTION_MAX_RETRIES,
    DEFAULT_BOT_DETECTION_RETRY_WAIT_TIME,
    DEFAULT_BOT_DETECTION_WAIT_FOR_PRESENCE_DELAY,
)


class BaseAutomationProcess:
    """Classe base para processos de automação web com ciclo de vida gerenciado.

    Gerencia o ciclo completo de uma automação: instanciação do handler,
    inicialização do driver, execução e encerramento. Suporta uso como
    context manager (``with`` statement) e como callable direto.

    Pipeline de execução ao chamar a instância:

    1. ``__enter__`` → :meth:`start` (se ``start=True``)
    2. :meth:`pre_run` (hook antes da execução)
    3. :meth:`run_executor` → :meth:`run` (lógica principal)
    4. :meth:`post_run` (hook após a execução)
    5. ``__exit__`` → :meth:`quit` (se ``quit=True``)

    Class Attributes:
        NAME: Nome único da automação. **Obrigatório** — definido na subclasse.
        WEB_DRIVER_HANDLER_CLASS: Classe do handler a instanciar. Padrão: :class:`ChromeWebDriverHandler`.
        BOT_DETECTION_METHODS: Lista de detectores de bot. Padrão: lista vazia.
        BOT_DETECTION_MAX_RETRIES: Máximo de retries ao detectar bot. Padrão: ``3``.
        BOT_DETECTION_RETRY_WAIT_TIME: Espera em segundos entre retries. Padrão: ``20``.
        BOT_DETECTION_WAIT_FOR_PRESENCE_DELAY: Timeout para detectar bot. Padrão: ``1``.

    Args:
        driver: Driver Selenium pré-criado. Se ``None``, o handler cria um.
        LOGGER: Logger da drtools. Usa logger padrão se não informado.
        start: Se ``True``, chama :meth:`start` ao entrar no context manager.
        quit: Se ``True``, chama :meth:`quit` ao sair do context manager.

    Raises:
        AssertionError: Se :attr:`NAME` não estiver definido.

    Example:
        >>> from drtools.web_automation.automation import BaseAutomationProcess
        >>> from drtools.web_automation.driver_handler.handler import WebDriverHandler
        ...
        >>> class ScrapePriceAutomation(BaseAutomationProcess):
        ...     NAME = "scrape-price"
        ...
        ...     def run(self, web_driver_handler: WebDriverHandler, url: str) -> dict:
        ...         web_driver_handler.go_to_page(url)
        ...         price_el = web_driver_handler.find_element('//span[@class="price"]')
        ...         return {"price": price_el.text if price_el else None}
        ...
        >>> automation = ScrapePriceAutomation(start=True, quit=True)
        >>> automation.start(remove_ui=True, load_images=False)
        >>> automation("https://example.com/product/1")
        >>> result = automation.get_result()
        >>> print(result["result"])  # {"price": "$29.99"}
    """

    NAME: str = None
    WEB_DRIVER_HANDLER_CLASS: Union[ChromeWebDriverHandler] = ChromeWebDriverHandler
    BOT_DETECTION_METHODS: List[BotDetection] = DEFAULT_BOT_DETECTION_METHODS
    BOT_DETECTION_MAX_RETRIES: int = DEFAULT_BOT_DETECTION_MAX_RETRIES
    BOT_DETECTION_RETRY_WAIT_TIME: int = DEFAULT_BOT_DETECTION_RETRY_WAIT_TIME
    BOT_DETECTION_WAIT_FOR_PRESENCE_DELAY: int = DEFAULT_BOT_DETECTION_WAIT_FOR_PRESENCE_DELAY

    def get_unique_id(self, starts_with: str='', ends_with: str='') -> str:
        """Gera um UUID único com prefixo e sufixo opcionais.

        Args:
            starts_with: Prefixo da string gerada.
            ends_with: Sufixo da string gerada.

        Returns:
            String no formato ``"{starts_with}{uuid4}{ends_with}"``.
        """
        return f'{starts_with}{str(uuid.uuid4())}{ends_with}'

    def __init__(
        self,
        driver: WebDriver=None,
        LOGGER: Logger=None,
        start: bool=False,
        quit: bool=False,
    ) -> None:
        assert self.NAME is not None, "NAME must be set."
        self.web_driver_handler = None
        if not LOGGER:
            LOGGER = Logger(
                name="BaseAutomationProcess",
                formatter_options=FormatterOptions(include_datetime=True, include_logger_name=True, include_level_name=True),
                default_start=False
            )
        self.driver = driver
        self.LOGGER = LOGGER
        self._start = start
        self._quit = quit
        self._result = None
        self.web_driver_handler = None
        self.web_driver_handler_start_kwargs = {}
        self.web_driver_start_args = ()
        self.web_driver_start_kwargs = {}
        self.start_web_driver_handler(self.driver, self.LOGGER)

    def set_driver(self, driver: WebDriver) -> None:
        """Substitui o driver no handler e no processo.

        Args:
            driver: Nova instância de WebDriver.
        """
        self.driver = driver
        self.web_driver_handler.set_driver(driver)

    def set_logger(self, LOGGER: Logger) -> None:
        """Substitui o logger no handler e no processo.

        Args:
            LOGGER: Novo logger da drtools.
        """
        self.LOGGER = LOGGER
        self.web_driver_handler.set_logger(LOGGER)

    def start_web_driver_handler(self, driver: WebDriver=None, LOGGER: Logger=None, **kwargs) -> None:
        """Instancia o :attr:`WEB_DRIVER_HANDLER_CLASS` com as configurações de bot detection.

        Chamado automaticamente no ``__init__``. Preserva os kwargs em
        :attr:`web_driver_handler_start_kwargs` para uso posterior (ex: cópia do handler).

        Args:
            driver: Driver a repassar ao handler.
            LOGGER: Logger a repassar ao handler.
            **kwargs: Configurações adicionais de bot detection (sobrescrevem os defaults).
        """
        kwargs.pop('driver', None)
        kwargs.pop('LOGGER', None)
        kwargs['bot_detection_methods'] = kwargs.get('bot_detection_methods', self.BOT_DETECTION_METHODS)
        kwargs['bot_detection_max_retries'] = kwargs.get('bot_detection_max_retries', self.BOT_DETECTION_MAX_RETRIES)
        kwargs['bot_detection_retry_wait_time'] = kwargs.get('bot_detection_retry_wait_time', self.BOT_DETECTION_RETRY_WAIT_TIME)
        kwargs['bot_detection_wait_for_presence_delay'] = kwargs.get('bot_detection_wait_for_presence_delay', self.BOT_DETECTION_WAIT_FOR_PRESENCE_DELAY)
        self.web_driver_handler_start_kwargs = deepcopy(kwargs)
        self.web_driver_handler = self.WEB_DRIVER_HANDLER_CLASS(driver, LOGGER, **kwargs)
        if driver:
            self.set_driver(driver)
        if LOGGER:
            self.set_logger(LOGGER)

    @property
    def web_driver_handler_start_args(self) -> Tuple:
        """Argumentos posicionais usados para inicializar o handler (driver e LOGGER)."""
        return (self.driver, self.LOGGER)

    def get_web_driver_handler_copy(self) -> WebDriverHandler:
        """Retorna uma cópia profunda do handler atual (útil para execução paralela).

        Returns:
            Nova instância de :class:`~drtools.web_automation.driver_handler.handler.WebDriverHandler`
            com a mesma configuração do handler original.
        """
        return deepcopy(self.web_driver_handler)

    def start(self, *args, **kwargs) -> None:
        """Inicializa o driver do handler com os argumentos fornecidos.

        Os argumentos são armazenados em :attr:`web_driver_start_args` e
        :attr:`web_driver_start_kwargs` para uso posterior (ex: criar novos handlers
        com a mesma config em :class:`BaseAutomationProcessFromList`).

        Args:
            *args: Repassados ao método ``start`` do handler.
            **kwargs: Repassados ao método ``start`` do handler (e.g. ``remove_ui``,
                ``load_images``, ``download_path``).

        Example:
            >>> automation.start(remove_ui=True, load_images=False, download_path="/tmp")
        """
        self.LOGGER.info(f"Initializing driver...")
        self.web_driver_start_args = deepcopy(args)
        self.web_driver_start_kwargs = deepcopy(kwargs)
        self.web_driver_handler.start(*args, **kwargs)
        self.LOGGER.info("Initializing driver... Done!")

    def quit(self):
        """Encerra o driver do handler.

        Example:
            >>> automation.quit()
        """
        self.LOGGER.info(f"Quiting driver...")
        self.web_driver_handler.quit()
        self.LOGGER.info(f"Quiting driver... Done!")

    def __enter__(self):
        """Inicia o driver ao entrar no context manager (se ``start=True``)."""
        if self._start:
            self.start()

    def __exit__(self, *args):
        """Encerra o driver ao sair do context manager (se ``quit=True``)."""
        if self._quit:
            self.quit()

    def get_result(self) -> AutomationResult:
        """Retorna o resultado da última execução.

        Returns:
            :class:`~drtools.web_automation.types.AutomationResult` ou ``None``
            se a automação ainda não foi executada.
        """
        return self._result

    def set_result(self, result: AutomationResult) -> None:
        """Define o resultado da execução.

        Args:
            result: :class:`~drtools.web_automation.types.AutomationResult` a armazenar.
        """
        self._result = result

    def get_execution_id(self) -> str:
        """Retorna o ID único da execução atual."""
        return self._execution_id

    def set_execution_id(self, execution_id: str) -> None:
        """Define o ID de execução.

        Args:
            execution_id: String identificadora da execução.
        """
        self._execution_id = execution_id

    def __call__(self, *args, **kwargs) -> None:
        """Executa o pipeline completo da automação como callable.

        Sequência: ``__enter__`` → ``pre_run`` → ``run_executor`` → ``set_result``
        → ``post_run`` → ``__exit__``.

        Args:
            *args: Repassados a ``pre_run``, ``run_executor`` e ``post_run``.
            **kwargs: Repassados a ``pre_run``, ``run_executor`` e ``post_run``.

        Example:
            >>> automation("https://example.com/product/1")
            >>> result = automation.get_result()
        """
        with self:
            started_at = datetime.now()
            self.set_execution_id(self.get_unique_id())
            self.pre_run(*args, **kwargs)
            automation_result: Any = self.run_executor(*args, **kwargs)
            self.set_result(
                AutomationResult(
                    execution_id=self.get_execution_id(),
                    started_at=str(started_at),
                    finished_at=str(datetime.now()),
                    result=automation_result,
                    extra=None,
                )
            )
            self.post_run(*args, **kwargs)

    def pre_run(self, *args, **kwargs) -> Any:
        """Hook executado antes de :meth:`run_executor`. Não-op por padrão.

        Sobrescreva para preparar estado, configurar headers, abrir sessões, etc.
        """
        pass

    def run_executor(self, *args, **kwargs) -> Any:
        """Delega a execução para :meth:`run` passando o handler como primeiro argumento.

        Returns:
            Valor retornado por :meth:`run`.
        """
        return self.run(self.web_driver_handler, *args, **kwargs)

    def run(self, web_driver_handler: WebDriverHandler, *args, **kwargs) -> Any:
        """Lógica principal da automação. **Deve ser implementado.**

        Args:
            web_driver_handler: Handler com o driver ativo.
            *args: Argumentos adicionais.
            **kwargs: Keyword arguments adicionais.

        Raises:
            NotImplementedError: Sempre que não for sobrescrito.
        """
        raise NotImplementedError

    def post_run(self, *args, **kwargs) -> Any:
        """Hook executado após :meth:`run_executor`. Não-op por padrão.

        Sobrescreva para salvar resultados, enviar notificações, liberar recursos, etc.
        """
        pass


class GoogleDriveUploadResults:
    """Utilitário para upload automático de resultados de automação para o Google Drive.

    Usado por :class:`GoogleDriveAutomationProcess` e
    :class:`GoogleDriveAutomationProcessFromList` em seus hooks ``post_run``.

    Args:
        automation: Instância da automação cujos resultados serão enviados.
        google_drive_base_folder_path: Caminho base no Drive onde os resultados
            serão salvos. Se ``None``, usa ``automation.GOOGLE_DRIVE_BASE_FOLDER_PATH``.

    Raises:
        Exception: Se nem ``google_drive_base_folder_path`` nem
            ``automation.GOOGLE_DRIVE_BASE_FOLDER_PATH`` estiverem definidos.

    Example:
        >>> uploader = GoogleDriveUploadResults(
        ...     automation=my_automation,
        ...     google_drive_base_folder_path="AutomationResults"
        ... )
        >>> uploader.set_credentials("service_account.json")
        >>> uploader.upload_to_google_drive()
    """

    def __init__(
        self,
        automation: BaseAutomationProcess,
        google_drive_base_folder_path: str=None,
    ) -> None:
        self._automation = automation
        self.credentials_filename = None
        self.service = None
        if not google_drive_base_folder_path:
            if not self._automation.GOOGLE_DRIVE_BASE_FOLDER_PATH:
                raise Exception("If google_drive_base_folder_path is not provided, GOOGLE_DRIVE_BASE_FOLDER_PATH must be set.")
            google_drive_base_folder_path = self._automation.GOOGLE_DRIVE_BASE_FOLDER_PATH
        if google_drive_base_folder_path.startswith('/'):
            google_drive_base_folder_path = google_drive_base_folder_path[1:]
        self.base_folder_path = google_drive_base_folder_path

    @property
    def results_folder(self) -> str:
        """Caminho da pasta de resultados no Drive: ``"{base_folder_path}/{automation.NAME}"``."""
        return f'{self.base_folder_path}/{self._automation.NAME}'

    def set_credentials(self, filename: str) -> None:
        """Define o arquivo de credenciais de service account para o Drive.

        Args:
            filename: Caminho para o arquivo JSON de service account.
        """
        self.credentials_filename = filename

    def start_service(self) -> None:
        """Constrói o serviço do Google Drive e resolve o ID da pasta base.

        Internamente cria um :class:`~drtools.google.drive.drive.DriveFromServiceAcountFile`
        e chama :meth:`~drtools.google.drive.drive.DriveFromServiceAcountFile.build`.
        """
        self.service = DriveFromServiceAcountFile(self.credentials_filename, LOGGER=self._automation.LOGGER)
        self.service.build()
        self.base_folder_id = self.service.get_folder_id_from_path(self.base_folder_path)

    def upload_to_google_drive(self):
        """Faz upload dos resultados da automação para o Google Drive como JSON.

        Cria (se necessário) a pasta ``{base_folder_path}/{automation.NAME}``
        e salva o resultado como JSON com nome
        ``created_at={timestamp}&execution_id={id}.json``.
        """
        self._automation.LOGGER.info('Uploading to Google Drive...')
        self.start_service()
        results = self._automation.get_result()
        execution_id = self._automation.get_execution_id()
        self.service.create_folder(self._automation.NAME, self.base_folder_id)
        timestamp = int(datetime.now().timestamp())
        self.service.upload_dict(
            results,
            f'{self.results_folder}/created_at={timestamp}&execution_id={execution_id}.json'
        )
        self._automation.LOGGER.info('Uploading to Google Drive... Done!')


class GoogleDriveAutomationProcess(BaseAutomationProcess):
    """Automação que salva os resultados automaticamente no Google Drive após a execução.

    Estende :class:`BaseAutomationProcess` adicionando upload do resultado para
    o Drive no hook ``post_run``.

    Class Attributes:
        GOOGLE_DRIVE_BASE_FOLDER_PATH: Caminho base no Drive para salvar resultados.
            Pode ser sobrescrito por ``google_drive_base_folder_path`` no ``__init__``.

    Args:
        google_drive_base_folder_path: Caminho base no Drive (sobrescreve :attr:`GOOGLE_DRIVE_BASE_FOLDER_PATH`).
        ignore_google_drive_savement: Se ``True``, desabilita o upload no ``post_run``.

    Example:
        >>> class MyAutomation(GoogleDriveAutomationProcess):
        ...     NAME = "my-automation"
        ...     GOOGLE_DRIVE_BASE_FOLDER_PATH = "AutomationResults"
        ...
        ...     def run(self, web_driver_handler, url):
        ...         web_driver_handler.go_to_page(url)
        ...         return {"title": web_driver_handler.driver.title}
        ...
        >>> automation = MyAutomation(start=True, quit=True)
        >>> automation.gdrive.set_credentials("service_account.json")
        >>> automation.start(remove_ui=True)
        >>> automation("https://example.com")
        >>> # Resultado salvo automaticamente no Drive
    """

    GOOGLE_DRIVE_BASE_FOLDER_PATH: str = None

    def __init__(
        self,
        *args,
        google_drive_base_folder_path: str=None,
        ignore_google_drive_savement: bool=False,
        **kwargs
    ) -> None:
        super(GoogleDriveAutomationProcess, self).__init__(*args, **kwargs)
        self.gdrive = GoogleDriveUploadResults(self, google_drive_base_folder_path)
        self._ignore_google_drive_savement = ignore_google_drive_savement

    def post_run(self, *args, **kwargs) -> Any:
        """Faz upload dos resultados para o Drive após a execução (a menos que ignorado)."""
        if not self._ignore_google_drive_savement:
            self.gdrive.upload_to_google_drive()


class BaseAutomationProcessFromList(BaseAutomationProcess):
    """Automação que processa uma lista de itens com suporte a paralelismo e retry.

    Estende :class:`BaseAutomationProcess` para iterar sobre uma lista de itens,
    processando cada um com o método :meth:`run`. Suporta:

    - Execução sequencial (``max_workers=1``) ou paralela (``max_workers > 1``).
    - Múltiplos drivers Chrome em paralelo (um por worker).
    - Retry automático por item com hooks pré/pós espera.
    - Pausas por bulk (``bulk_size``) e entre itens (``wait_time``).
    - Rastreamento de sucesso/erro por driver e por execução.

    Class Attributes:
        AUTOMATION_RESULTS_EXECUTION_ARCHIVE_KEY: Chave usada internamente para
            armazenar o resultado acumulado como atributo da instância.

    Args:
        LOGGER: Logger da drtools.
        start: Se ``True``, inicia os drivers ao entrar no context manager.
        quit: Se ``True``, encerra os drivers ao sair do context manager.
        raise_exception: Se ``True``, propaga exceções de itens individuais
            interrompendo o processamento. Padrão: ``False``.
        bulk_size: Número de itens após o qual os hooks ``wait_pre_action`` e
            ``wait_post_action`` são chamados. ``None`` desabilita.
        wait_time: Segundos de espera entre o processamento de cada item. ``None`` desabilita.
        verbose_traceback: Se ``True``, loga o traceback completo em caso de erro.
        max_workers: Número de threads/drivers paralelos. Padrão: ``1`` (sequencial).
        worker_max_tries: Número máximo de tentativas por item. Padrão: ``3``.
        retry_wait_time: Segundos de espera entre retries de um item. Padrão: ``30``.

    Example:
        >>> class ScrapeProductList(BaseAutomationProcessFromList):
        ...     NAME = "scrape-products"
        ...
        ...     def run(
        ...         self,
        ...         web_driver_handler: WebDriverHandler,
        ...         product_url: str,
        ...         list_item_idx: int,
        ...     ) -> dict:
        ...         web_driver_handler.go_to_page(product_url)
        ...         price = web_driver_handler.find_element('//span[@class="price"]')
        ...         return {"url": product_url, "price": price.text if price else None}
        ...
        >>> automation = ScrapeProductList(
        ...     max_workers=3,
        ...     worker_max_tries=2,
        ...     wait_time=1,
        ...     raise_exception=False,
        ... )
        >>> automation.start(remove_ui=True, load_images=False, download_path="/tmp")
        >>> urls = ["https://example.com/p/1", "https://example.com/p/2"]
        >>> automation(urls)
        >>> result = automation.get_result()
        >>> print(result["result"]["success_rate"])  # 0.95
    """

    AUTOMATION_RESULTS_EXECUTION_ARCHIVE_KEY: str = "_automation_results"

    def __init__(
        self,
        LOGGER: Logger=None,
        start: bool=False,
        quit: bool=False,
        raise_exception: bool=False,
        bulk_size: int=None,
        wait_time: int=None,
        verbose_traceback: bool=False,
        max_workers: int=1,
        worker_max_tries: int=3,
        retry_wait_time: int=30,
    ) -> None:
        super(BaseAutomationProcessFromList, self).__init__(None, LOGGER, start, quit)
        self.raise_exception = raise_exception
        self.bulk_size = bulk_size
        self.wait_time = wait_time
        self.verbose_traceback = verbose_traceback
        self.max_workers = max_workers
        self.worker_max_tries = worker_max_tries
        self.retry_wait_time = retry_wait_time
        self._web_driver_handlers = []
        self._lock = Lock()
        self._success_executions_by_handler = {}
        self._errors_executions_by_handler = {}

    def add_web_driver_handler(self, web_driver_handler: WebDriverHandler) -> None:
        """Registra um novo handler na pool de drivers disponíveis.

        Thread-safe. Inicializa os contadores de sucesso e erro para o handler.

        Args:
            web_driver_handler: Handler a adicionar.
        """
        with self._lock:
            self._web_driver_handlers.append(web_driver_handler)
            self._success_executions_by_handler[web_driver_handler] = 0
            self._errors_executions_by_handler[web_driver_handler] = 0

    def start(self, *args, **kwargs):
        """Inicializa um driver Chrome para cada worker (``max_workers`` drivers no total).

        Cria cópias do handler base e inicia cada uma com os argumentos fornecidos.
        Quando ``max_workers > 1`` e ``download_path`` é fornecido, sufixos
        numéricos são adicionados ao caminho (``download_path-1``, ``download_path-2``, ...).

        Args:
            *args: Repassados ao ``start`` de cada handler.
            **kwargs: Repassados ao ``start`` de cada handler.
        """
        self.LOGGER.info(f"Initializing drivers...")
        self.web_driver_start_args = deepcopy(args)
        self.web_driver_start_kwargs = deepcopy(kwargs)
        for i in range(self.max_workers):
            web_driver_handler = self.get_web_driver_handler_copy()
            cp_kwargs = deepcopy(kwargs)
            if cp_kwargs.get('download_path', False) and self.max_workers > 1:
                cp_kwargs['download_path'] = f"{cp_kwargs['download_path']}-{i+1}"
            web_driver_handler.start(*args, **cp_kwargs)
            os.makedirs(web_driver_handler.download_path)
            self.add_web_driver_handler(web_driver_handler)
        self.LOGGER.info("Initializing drivers... Done!")

    def quit(self):
        """Encerra todos os drivers da pool."""
        self.LOGGER.info(f"Quiting drivers...")
        for web_driver_handler in self._web_driver_handlers:
            web_driver_handler.quit()
        self.LOGGER.info(f"Quiting drivers... Done!")

    def __call__(self, list_items: List[Any], *args, **kwargs) -> None:
        """Executa a automação sobre uma lista de itens.

        Args:
            list_items: Lista de itens a processar.
            *args: Repassados ao método :meth:`run` de cada item.
            **kwargs: Repassados ao método :meth:`run` de cada item.
        """
        return super(BaseAutomationProcessFromList, self).__call__(list_items, *args, **kwargs)

    def run_executor(self, list_items: List[Any], *args, **kwargs) -> AutomationFromListResult:
        """Orquestra o processamento da lista e compila as estatísticas finais.

        Returns:
            :class:`~drtools.web_automation.types.AutomationFromListResult` com
            contadores de sucesso/erro, taxa de sucesso e resultados individuais.
        """
        self.initialize_automation_result_value()
        total = len(list_items)
        started_at = datetime.now()
        self.process_list_items(list_items, started_at, total, *args, **kwargs)
        error_count = self.get_automation_error_count()
        success_count = self.get_automation_success_count()
        self.set_automation_success_rate(success_count/(success_count+error_count))
        success_rate = round(100*self.get_automation_success_rate(), 2)
        self.LOGGER.info(f"Automation completed with {success_rate}% success rate.")
        return self.get_automation_result()

    def process_list_items(self, list_items: List[Any], started_at: datetime, total: int, *args, **kwargs) -> None:
        """Despacha o processamento para sequencial ou paralelo conforme ``max_workers``.

        Args:
            list_items: Lista de itens a processar.
            started_at: Timestamp de início para cálculo de tempo restante.
            total: Total de itens (para log de progresso).
        """
        if self.max_workers == 1:
            return self.sequential_list_processing(list_items, started_at, total, *args, **kwargs)
        else:
            return self.threading_list_processing(list_items, started_at, total, *args, **kwargs)

    def sequential_list_processing(self, list_items: List[Any], started_at: datetime, total: int, *args, **kwargs) -> None:
        """Processa a lista de forma sequencial (um item por vez)."""
        for list_item_idx, list_item in enumerate(list_items):
            self.run_middleware(
                Worker(
                    list_item=list_item,
                    list_item_idx=list_item_idx,
                    started_at=started_at,
                    total=total,
                    args=args,
                    kwargs=kwargs
                )
            )

    def threading_list_processing(self, list_items: List[Any], started_at: datetime, total: int, *args, **kwargs) -> None:
        """Processa a lista em paralelo usando ``ThreadPoolExecutor``."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.run_middleware,
                    Worker(
                        list_item=list_item,
                        list_item_idx=list_item_idx,
                        started_at=started_at,
                        total=total,
                        args=args,
                        kwargs=kwargs
                    )
                ): list_item
                for list_item_idx, list_item in enumerate(list_items)
            }
            for future in futures_as_completed(futures):
                future.result()

    def run_middleware(self, worker: Worker) -> None:
        """Middleware de execução de um item individual.

        Responsável por:
        - Retirar um handler da pool (thread-safe).
        - Executar :meth:`run` com retry via :func:`drtools.utils.retry`.
        - Registrar sucesso ou erro no resultado consolidado.
        - Logar progresso e tempo restante estimado.
        - Aplicar ``wait_time`` e hooks ``bulk_size`` quando configurados.
        - Devolver o handler à pool após a execução.

        Args:
            worker: :class:`~drtools.web_automation.types.Worker` com os dados do item.
        """
        single_execution_id = self.get_unique_id()
        single_execution_id_label = f'AutomationWokerID:{single_execution_id}'
        def _msg_id(msg: str, remove_bk: bool=True):
            _msg = msg
            if remove_bk:
                _msg = remove_break_line(str(_msg))
            return f'[{single_execution_id_label}] {_msg}'
        list_item = worker['list_item']
        list_item_cp = deepcopy(list_item)
        list_item_idx = worker['list_item_idx']
        started_at = worker['started_at']
        total = worker['total']
        args = worker['args']
        kwargs = worker['kwargs']
        list_item_result = None
        error = None
        error_traceback = None
        item_started_at = datetime.now()
        web_driver_handler = self.pop_web_driver_handler()
        try:
            list_item_result, last_exception = retry(
                func=self.run,
                func_args=(web_driver_handler, list_item, list_item_idx, *args),
                func_kwargs=kwargs,
                pre_wait_retry=self.retry_pre_wait_action,
                pre_wait_retry_args=(web_driver_handler, list_item, list_item_idx, *args),
                pre_wait_retry_kwargs=kwargs,
                post_wait_retry=self.retry_post_wait_action,
                post_wait_retry_args=(web_driver_handler, list_item, list_item_idx, *args),
                post_wait_retry_kwargs=kwargs,
                LOGGER=self.LOGGER,
                raise_exception=True,
                wait_time=self.retry_wait_time,
                max_tries=self.worker_max_tries,
                execution_id=single_execution_id_label,
                verbose_traceback=self.verbose_traceback,
            )
            self.increment_automation_success_count()
            self.increment_web_driver_handler_success_count(web_driver_handler)
        except Exception as exc:
            if self.raise_exception:
                raise exc
            error = str(exc)
            error_traceback = traceback.format_exc()
            self.LOGGER.error(_msg_id(f'Error: {error}'))
            if self.verbose_traceback:
                self.LOGGER.error(_msg_id(error_traceback, remove_bk=False))
            self.increment_automation_error_count()
            self.increment_web_driver_handler_error_count(web_driver_handler)
        self.append_automation_result(
                AutomationFromListItemResult(
                id=single_execution_id,
                started_at=str(item_started_at),
                finished_at=str(datetime.now()),
                error=error,
                error_traceback=error_traceback,
                list_item_result=list_item_result,
                list_item=list_item_cp,
            )
        )
        processed_items_num = self.get_automation_processed_items_num()
        total_time = (datetime.now()-started_at).total_seconds()
        speed = total_time / processed_items_num
        remaining_time = (total - processed_items_num) * speed
        remaining_time_msg = display_time(int(remaining_time))
        success_count = self.get_automation_success_count()
        error_count = self.get_automation_error_count()
        self.LOGGER.debug(_msg_id(
            f"(C: {processed_items_num:,} | T: {total:,} | S: {success_count:,} | E: {error_count:,}) Complete! Expected remaining time: {remaining_time_msg}..."
        ))
        if self.bulk_size:
            processed_items_num = self.get_web_driver_handler_execution_count(web_driver_handler)
            if processed_items_num % self.bulk_size == 0:
                self.LOGGER.debug(_msg_id(f'Waiting pre action...'))
                self.wait_pre_action()
                self.LOGGER.debug(_msg_id(f'Waiting pre action... Done!'))
        if self.wait_time:
            self.LOGGER.debug(_msg_id(f'Waiting for {self.wait_time:,}s...'))
            time.sleep(self.wait_time)
            self.LOGGER.debug(_msg_id(f'Waiting for {self.wait_time:,}s... Done!'))
        if self.bulk_size:
            processed_items_num = self.get_web_driver_handler_execution_count(web_driver_handler)
            if processed_items_num % self.bulk_size == 0:
                self.LOGGER.debug(_msg_id(f'Waiting post action...'))
                self.wait_post_action()
                self.LOGGER.debug(_msg_id(f'Waiting post action... Done!'))
        self.handle_web_driver_handler_after_run(web_driver_handler)

    # --- Driver pool management ---

    def pop_web_driver_handler(self) -> WebDriverHandler:
        """Retira o primeiro handler disponível da pool (thread-safe).

        Returns:
            :class:`~drtools.web_automation.driver_handler.handler.WebDriverHandler`
            disponível para uso.
        """
        with self._lock:
            web_driver_handler: WebDriverHandler = self._web_driver_handlers.pop(0)
            return web_driver_handler

    def handle_web_driver_handler_after_run(self, web_driver_handler: WebDriverHandler) -> None:
        """Devolve o handler à pool após o processamento de um item (thread-safe).

        Args:
            web_driver_handler: Handler a devolver.
        """
        with self._lock:
            self._web_driver_handlers.append(web_driver_handler)

    def get_web_driver_handlers(self) -> List[WebDriverHandler]:
        """Retorna a lista atual de handlers na pool."""
        return self._web_driver_handlers

    def get_web_driver_handler_execution_count(self, web_driver_handler: WebDriverHandler) -> int:
        """Retorna o total de execuções (sucesso + erro) de um handler específico.

        Args:
            web_driver_handler: Handler a consultar.
        """
        return self._success_executions_by_handler[web_driver_handler] + self._errors_executions_by_handler[web_driver_handler]

    def increment_web_driver_handler_success_count(self, web_driver_handler: WebDriverHandler) -> None:
        """Incrementa o contador de sucesso de um handler (thread-safe)."""
        with self._lock:
            self._success_executions_by_handler[web_driver_handler] += 1

    def increment_web_driver_handler_error_count(self, web_driver_handler: WebDriverHandler) -> None:
        """Incrementa o contador de erro de um handler (thread-safe)."""
        with self._lock:
            self._errors_executions_by_handler[web_driver_handler] += 1

    # --- Automation result management ---

    def initialize_automation_result_value(self) -> None:
        """Inicializa o resultado acumulado da automação com zeros."""
        setattr(
            self,
            self.AUTOMATION_RESULTS_EXECUTION_ARCHIVE_KEY,
            AutomationFromListResult(
                success_count=0,
                error_count=0,
                success_rate=None,
                automation_results=[]
            )
        )

    def get_automation_result(self) -> AutomationFromListResult:
        """Retorna o resultado acumulado da automação."""
        return getattr(self, self.AUTOMATION_RESULTS_EXECUTION_ARCHIVE_KEY)

    def set_automation_success_rate(self, success_rate: float) -> None:
        """Define a taxa de sucesso no resultado acumulado."""
        automation_result = self.get_automation_result()
        automation_result['success_rate'] = success_rate

    def get_automation_success_rate(self) -> float:
        """Retorna a taxa de sucesso atual (0.0 a 1.0)."""
        automation_result = self.get_automation_result()
        return automation_result['success_rate']

    def get_automation_success_count(self) -> int:
        """Retorna o número de itens processados com sucesso."""
        automation_result = self.get_automation_result()
        return automation_result['success_count']

    def get_automation_error_count(self) -> int:
        """Retorna o número de itens que falharam."""
        automation_result = self.get_automation_result()
        return automation_result['error_count']

    def increment_automation_success_count(self) -> None:
        """Incrementa o contador de sucesso global (thread-safe)."""
        with self._lock:
            automation_result = self.get_automation_result()
            automation_result['success_count'] += 1

    def increment_automation_error_count(self) -> None:
        """Incrementa o contador de erro global (thread-safe)."""
        with self._lock:
            automation_result = self.get_automation_result()
            automation_result['error_count'] += 1

    def append_automation_result(self, automation_result_item: AutomationFromListItemResult) -> None:
        """Adiciona o resultado de um item individual ao resultado acumulado (thread-safe)."""
        with self._lock:
            automation_results = self.get_automation_result()
            automation_results['automation_results'].append(automation_result_item)

    def get_automation_processed_items_num(self) -> int:
        """Retorna o total de itens processados até o momento (sucesso + erro)."""
        automation_result = self.get_automation_result()
        return automation_result['success_count'] + automation_result['error_count']

    # --- Retry hooks ---

    def retry_pre_wait_action(
        self,
        last_exception: Exception,
        web_driver_handler: WebDriverHandler,
        list_item: Any,
        list_item_idx: int,
        *args,
        **kwargs
    ) -> Any:
        """Hook chamado **antes** de esperar entre retries de um item. Não-op por padrão.

        Sobrescreva para reiniciar o driver, limpar cookies, etc.

        Args:
            last_exception: Última exceção capturada.
            web_driver_handler: Handler em uso.
            list_item: Item que falhou.
            list_item_idx: Índice do item.
        """
        pass

    def retry_post_wait_action(
        self,
        last_exception: Exception,
        web_driver_handler: WebDriverHandler,
        list_item: Any,
        list_item_idx: int,
        *args,
        **kwargs
    ) -> Any:
        """Hook chamado **após** esperar entre retries de um item. Não-op por padrão.

        Args:
            last_exception: Última exceção capturada.
            web_driver_handler: Handler em uso.
            list_item: Item que falhou.
            list_item_idx: Índice do item.
        """
        pass

    def wait_pre_action(
        self,
        web_driver_handler: WebDriverHandler,
        list_item: Any,
        list_item_idx: int,
        *args,
        **kwargs
    ) -> Any:
        """Hook chamado **antes** da pausa de ``bulk_size``. Não-op por padrão.

        Sobrescreva para fechar popups, salvar estado parcial, etc.
        """
        pass

    def wait_post_action(
        self,
        web_driver_handler: WebDriverHandler,
        list_item: Any,
        list_item_idx: int,
        *args,
        **kwargs
    ) -> Any:
        """Hook chamado **após** a pausa de ``bulk_size``. Não-op por padrão."""
        pass

    def run(self, web_driver_handler: WebDriverHandler, list_item: Any, list_item_idx: int, *args, **kwargs) -> Any:
        """Lógica de processamento de um item individual. **Deve ser implementado.**

        Args:
            web_driver_handler: Handler com o driver ativo para este worker.
            list_item: Item a processar (URL, ID, dict, etc.).
            list_item_idx: Índice do item na lista original.
            *args: Argumentos extras passados ao ``__call__``.
            **kwargs: Keyword arguments extras.

        Raises:
            NotImplementedError: Sempre que não for sobrescrito.
        """
        raise NotImplementedError


class GoogleDriveAutomationProcessFromList(BaseAutomationProcessFromList):
    """Automação de lista que salva os resultados automaticamente no Google Drive.

    Estende :class:`BaseAutomationProcessFromList` com upload para o Drive
    no hook ``post_run``.

    Class Attributes:
        GOOGLE_DRIVE_BASE_FOLDER_PATH: Caminho base no Drive para salvar resultados.

    Args:
        google_drive_base_folder_path: Caminho base no Drive (sobrescreve o class attribute).
        ignore_google_drive_savement: Se ``True``, desabilita o upload no ``post_run``.

    Example:
        >>> class ScrapeProducts(GoogleDriveAutomationProcessFromList):
        ...     NAME = "scrape-products"
        ...     GOOGLE_DRIVE_BASE_FOLDER_PATH = "AutomationResults"
        ...
        ...     def run(self, web_driver_handler, product_url, list_item_idx):
        ...         web_driver_handler.go_to_page(product_url)
        ...         return {"title": web_driver_handler.driver.title}
        ...
        >>> automation = ScrapeProducts(max_workers=2)
        >>> automation.gdrive.set_credentials("service_account.json")
        >>> automation.start(remove_ui=True)
        >>> automation(["https://example.com/p/1", "https://example.com/p/2"])
        >>> # Resultados salvos no Drive em AutomationResults/scrape-products/
    """

    GOOGLE_DRIVE_BASE_FOLDER_PATH: str = None

    def __init__(
        self,
        *args,
        google_drive_base_folder_path: str=None,
        ignore_google_drive_savement: bool=False,
        **kwargs
    ) -> None:
        super(GoogleDriveAutomationProcessFromList, self).__init__(*args, **kwargs)
        self.gdrive = GoogleDriveUploadResults(self, google_drive_base_folder_path)
        self._ignore_google_drive_savement = ignore_google_drive_savement

    def post_run(self, *args, **kwargs) -> Any:
        """Faz upload dos resultados para o Drive após processar toda a lista."""
        if not self._ignore_google_drive_savement:
            self.gdrive.upload_to_google_drive()


class ProxyAutomation(BaseAutomationProcessFromList):
    """Automação de lista com rotação de proxies por item processado.

    Cada item da lista recebe um driver Chrome criado com um proxy diferente,
    escolhido aleatoriamente de ``proxies``. O driver é encerrado após o
    processamento do item (não reutilizado).

    Args:
        proxies: Lista de URLs de proxy (e.g. ``["http://user:pass@host:port"]``).
        *args: Repassados a :class:`BaseAutomationProcessFromList`.
        **kwargs: Repassados a :class:`BaseAutomationProcessFromList`.

    Example:
        >>> class ScrapeWithProxy(ProxyAutomation):
        ...     NAME = "scrape-proxy"
        ...
        ...     def run(self, web_driver_handler, url, list_item_idx):
        ...         web_driver_handler.go_to_page(url)
        ...         return {"title": web_driver_handler.driver.title}
        ...
        >>> proxies = [
        ...     "http://user1:pass1@proxy1.example.com:8080",
        ...     "http://user2:pass2@proxy2.example.com:8080",
        ... ]
        >>> automation = ScrapeWithProxy(proxies=proxies)
        >>> automation.start(remove_ui=True)
        >>> automation(["https://example.com/p/1", "https://example.com/p/2"])
    """

    def __init__(self, *args, proxies: List[str], **kwargs) -> None:
        super(ProxyAutomation, self).__init__(*args, **kwargs)
        self._proxies = proxies

    def start(self, *args, **kwargs):
        """Armazena os argumentos de inicialização sem criar drivers (criados por demanda)."""
        self.web_driver_start_args = deepcopy(args)
        self.web_driver_start_kwargs = deepcopy(kwargs)

    def quit(self):
        """Não-op: drivers são encerrados individualmente após cada item."""
        pass

    def get_proxy_url(self) -> str:
        """Escolhe aleatoriamente um proxy da lista.

        Returns:
            URL de proxy selecionada.
        """
        return random.choice(self._proxies)

    def get_proxy_web_driver_handler(self) -> WebDriverHandler:
        """Cria um novo driver Chrome configurado com um proxy aleatório.

        Combina as opções do seleniumwire originais com as configurações do proxy
        selecionado e inicia um novo handler.

        Returns:
            Handler com driver Chrome configurado com proxy.
        """
        proxy_url = self.get_proxy_url()
        seleniumwire_options = {'proxy': {'http': f'{proxy_url}', 'https': f'{proxy_url}','verify_ssl': False}}
        original_seleniumwire_options = self.web_driver_start_kwargs.pop('seleniumwire_options', {})
        seleniumwire_options = {**original_seleniumwire_options, **seleniumwire_options}
        kwargs = {**self.web_driver_start_kwargs, 'seleniumwire_options': seleniumwire_options}
        web_driver_handler: WebDriverHandler = self.get_web_driver_handler_copy()
        web_driver_handler.start(*self.web_driver_start_args, **kwargs)
        self.add_web_driver_handler(web_driver_handler)
        return web_driver_handler

    def pop_web_driver_handler(self) -> WebDriverHandler:
        """Cria e retorna um novo handler com proxy ao invés de retirar da pool.

        Returns:
            Novo handler com driver Chrome e proxy configurado.
        """
        return self.get_proxy_web_driver_handler()

    def handle_web_driver_handler_after_run(self, web_driver_handler: WebDriverHandler) -> None:
        """Encerra o driver e remove o handler da pool após o processamento do item.

        Args:
            web_driver_handler: Handler a encerrar e remover.

        Raises:
            Exception: Se o handler não for encontrado na pool.
        """
        web_driver_handler.quit()
        with self._lock:
            for i in range(len(self._web_driver_handlers)):
                wdh = self._web_driver_handlers[i]
                if web_driver_handler == wdh:
                    self._web_driver_handlers.pop(i)
                    del self._success_executions_by_handler[web_driver_handler]
                    del self._errors_executions_by_handler[web_driver_handler]
                    return None
        raise Exception(f"Web Driver Handler not found: {web_driver_handler}")
