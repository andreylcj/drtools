

from .driver_handler.handler import WebDriverHandler
from .driver_handler.chrome import ChromeWebDriverHandler
from typing import List, Any, Union, Tuple
from drtools.logging import Logger, FormatterOptions
from drtools.google.drive.drive import DriveFromServiceAcountFile
import uuid
from datetime import datetime
import time
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
    """
    - On self.__init__(), starts and Web Driver Handler instance from method self.start_web_driver_handler().
    - Method self.start_web_driver_handler() instantiate the Web Driver Handler instance defined on self.WEB_DRIVER_HANDLER_CLASS
    - Attribute self.web_driver_handler_start_kwargs is set on self.start_web_driver_handler()
    - When call self.start() method, will start web driver on Web Driver Handler instance initiated on self.__init__()
    """
    
    NAME: str=None # Automation name. Must be unique. Mandatory
    WEB_DRIVER_HANDLER_CLASS: Union[ChromeWebDriverHandler] = ChromeWebDriverHandler # Web Driver Handler instance to be initiated on self.start_web_driver_handler()
    BOT_DETECTION_METHODS: List[BotDetection] = DEFAULT_BOT_DETECTION_METHODS
    BOT_DETECTION_MAX_RETRIES: int = DEFAULT_BOT_DETECTION_MAX_RETRIES
    BOT_DETECTION_RETRY_WAIT_TIME: int = DEFAULT_BOT_DETECTION_RETRY_WAIT_TIME
    BOT_DETECTION_WAIT_FOR_PRESENCE_DELAY: int = DEFAULT_BOT_DETECTION_WAIT_FOR_PRESENCE_DELAY
    
    def get_unique_id(self, starts_with: str='', ends_with: str='') -> str:
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
        self.driver = driver
        self.web_driver_handler.set_driver(driver)
    
    def set_logger(self, LOGGER: Logger) -> None:
        self.LOGGER = LOGGER
        self.web_driver_handler.set_logger(LOGGER)
    
    def start_web_driver_handler(self, driver: WebDriver=None, LOGGER: Logger=None, **kwargs) -> None:
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
        return (self.driver, self.LOGGER)
    
    def get_web_driver_handler_copy(self) -> WebDriverHandler:
        return deepcopy(self.web_driver_handler)
    
    def start(self, *args, **kwargs) -> None:
        self.LOGGER.info(f"Initializing driver...")
        self.web_driver_start_args = deepcopy(args)
        self.web_driver_start_kwargs = deepcopy(kwargs)
        self.web_driver_handler.start(*args, **kwargs)
        self.LOGGER.info("Initializing driver... Done!")
    
    def quit(self):
        self.LOGGER.info(f"Quiting driver...")
        self.web_driver_handler.quit()
        self.LOGGER.info(f"Quiting driver... Done!")
    
    def __enter__(self):
        if self._start:
            self.start()
    
    def __exit__(self, *args):
        if self._quit:
            self.quit()
    
    def get_result(self) -> AutomationResult:
        return self._result
    
    def set_result(self, result: AutomationResult) -> None:
        self._result = result
    
    def get_execution_id(self) -> str:
        return self._execution_id
    
    def set_execution_id(self, execution_id: str) -> None:
        self._execution_id = execution_id
    
    def __call__(self, *args, **kwargs) -> None:
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
                    extra=None, # Set on post run if needed
                )
            )
            self.post_run(*args, **kwargs)
    
    def pre_run(self, *args, **kwargs) -> Any:
        pass
        
    def run_executor(self, *args, **kwargs) -> Any:
        return self.run(self.web_driver_handler, *args, **kwargs)
    
    def run(self, web_driver_handler: WebDriverHandler, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    def post_run(self, *args, **kwargs) -> Any:
        pass
    

class GoogleDriveUploadResults:
    
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
        self.base_folder_path = google_drive_base_folder_path # base folder path must exists on google drive
    
    @property
    def results_folder(self) -> str:
        return f'{self.base_folder_path}/{self._automation.NAME}'
    
    def set_credentials(self, filename: str) -> None:
        self.credentials_filename = filename
        
    def start_service(self) -> None:
        self.service = DriveFromServiceAcountFile(self.credentials_filename, LOGGER=self._automation.LOGGER)
        self.service.build()
        self.base_folder_id = self.service.get_folder_id_from_path(self.base_folder_path)
        
    def upload_to_google_drive(self):
        self._automation.LOGGER.info('Uploading to Google Drive...')
        self.start_service()
        results = self._automation.get_result()
        execution_id = self._automation.get_execution_id()
        # create folder if not exists
        self.service.create_folder(self._automation.NAME, self.base_folder_id)
        # upload result
        timestamp = int(datetime.now().timestamp())
        self.service.upload_dict(
            results, 
            f'{self.results_folder}/created_at={timestamp}&execution_id={execution_id}.json'
        )
        self._automation.LOGGER.info('Uploading to Google Drive... Done!')
    
    
class GoogleDriveAutomationProcess(BaseAutomationProcess):
    
    GOOGLE_DRIVE_BASE_FOLDER_PATH: str = None
    
    def __init__(self, *args, google_drive_base_folder_path: str=None, **kwargs) -> None:
        super(GoogleDriveAutomationProcess, self).__init__(*args, **kwargs)
        self.gdrive = GoogleDriveUploadResults(self, google_drive_base_folder_path)
    
    def post_run(self, *args, **kwargs) -> Any:
        self.gdrive.upload_to_google_drive()


class BaseAutomationProcessFromList(BaseAutomationProcess):
    
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
        self._success_executions_by_handler = {} # web_driver_handler -> execution_count
        self._errors_executions_by_handler = {} # web_driver_handler -> execution_count
    
    def add_web_driver_handler(self, web_driver_handler: WebDriverHandler) -> None:
        with self._lock:
            self._web_driver_handlers.append(web_driver_handler)
            self._success_executions_by_handler[web_driver_handler] = 0
            self._errors_executions_by_handler[web_driver_handler] = 0
    
    def start(self, *args, **kwargs):
        self.LOGGER.info(f"Initializing drivers...")
        self.web_driver_start_args = deepcopy(args)
        self.web_driver_start_kwargs = deepcopy(kwargs)
        for i in range(self.max_workers):
            web_driver_handler = self.get_web_driver_handler_copy()
            web_driver_handler.start(*args, **kwargs)
            self.add_web_driver_handler(web_driver_handler)
        self.LOGGER.info("Initializing drivers... Done!")
    
    def quit(self):
        self.LOGGER.info(f"Quiting drivers...")
        for web_driver_handler in self._web_driver_handlers:
            web_driver_handler.quit()
        self.LOGGER.info(f"Quiting drivers... Done!")
    
    def __call__(self, list_items: List[Any], *args, **kwargs) -> None:
        return super(BaseAutomationProcessFromList, self).__call__(list_items, *args, **kwargs)
    
    def run_executor(self, list_items: List[Any], *args, **kwargs) -> AutomationFromListResult:
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
        if self.max_workers == 1:
            return self.sequential_list_processing(list_items, started_at, total, *args, **kwargs)
        else:
            return self.threading_list_processing(list_items, started_at, total, *args, **kwargs)
    
    def sequential_list_processing(self, list_items: List[Any], started_at: datetime, total: int, *args, **kwargs) -> None:
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
        single_execution_id = self.get_unique_id()
        single_execution_id_label = f'AutomationWokerID:{single_execution_id}'
        def _msg_id(msg: str):
            return f'[{single_execution_id_label}] {remove_break_line(str(msg))}'
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
                self.LOGGER.error(_msg_id(error_traceback))
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
    
    #########################
    
    def pop_web_driver_handler(self) -> WebDriverHandler:
        with self._lock:
            web_driver_handler: WebDriverHandler = self._web_driver_handlers.pop(0)
            return web_driver_handler
    
    def handle_web_driver_handler_after_run(self, web_driver_handler: WebDriverHandler) -> None:
        with self._lock:
            self._web_driver_handlers.append(web_driver_handler)
    
    def get_web_driver_handlers(self) -> List[WebDriverHandler]:
        return self._web_driver_handlers
    
    def get_web_driver_handler_execution_count(self, web_driver_handler: WebDriverHandler) -> int:
        return self._success_executions_by_handler[web_driver_handler] + self._errors_executions_by_handler[web_driver_handler]
        
    def increment_web_driver_handler_success_count(self, web_driver_handler: WebDriverHandler) -> None:
        with self._lock:
            self._success_executions_by_handler[web_driver_handler] += 1
            
    def increment_web_driver_handler_error_count(self, web_driver_handler: WebDriverHandler) -> None:
        with self._lock:
            self._errors_executions_by_handler[web_driver_handler] += 1
    
    #########################
    
    def initialize_automation_result_value(self) -> None:
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
        return getattr(self, self.AUTOMATION_RESULTS_EXECUTION_ARCHIVE_KEY)
    
    def set_automation_success_rate(self, success_rate: float) -> None:
        automation_result = self.get_automation_result()
        automation_result['success_rate'] = success_rate
    
    def get_automation_success_rate(self) -> float:
        automation_result = self.get_automation_result()
        return automation_result['success_rate']
    
    def get_automation_success_count(self) -> int:
        automation_result = self.get_automation_result()
        return automation_result['success_count']
    
    def get_automation_error_count(self) -> int:
        automation_result = self.get_automation_result()
        return automation_result['error_count']
    
    def increment_automation_success_count(self) -> None:
        with self._lock:
            automation_result = self.get_automation_result()
            automation_result['success_count'] += 1
        
    def increment_automation_error_count(self) -> None:
        with self._lock:
            automation_result = self.get_automation_result()
            automation_result['error_count'] += 1
    
    def append_automation_result(self, automation_result_item: AutomationFromListItemResult) -> None:
        with self._lock:
            automation_results = self.get_automation_result()
            automation_results['automation_results'].append(automation_result_item)
        
    def get_automation_processed_items_num(self) -> int:
        automation_result = self.get_automation_result()
        return automation_result['success_count'] + automation_result['error_count']
    
    #########################
    
    def retry_pre_wait_action(
        self, 
        last_exception: Exception, 
        web_driver_handler: WebDriverHandler, 
        list_item: Any, 
        list_item_idx: int, 
        *args, 
        **kwargs
    ) -> Any:
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
        pass
    
    def wait_pre_action(
        self, 
        web_driver_handler: WebDriverHandler, 
        list_item: Any, 
        list_item_idx: int, 
        *args, 
        **kwargs
    ) -> Any:
        pass
    
    def wait_post_action(
        self, 
        web_driver_handler: WebDriverHandler, 
        list_item: Any, 
        list_item_idx: int, 
        *args, 
        **kwargs
    ) -> Any:
        pass
    
    def run(self, web_driver_handler: WebDriverHandler, list_item: Any, list_item_idx: int, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    
class GoogleDriveAutomationProcessFromList(BaseAutomationProcessFromList):
    
    GOOGLE_DRIVE_BASE_FOLDER_PATH: str = None
    
    def __init__(self, *args, google_drive_base_folder_path: str=None, **kwargs) -> None:
        super(GoogleDriveAutomationProcessFromList, self).__init__(*args, **kwargs)
        self.gdrive = GoogleDriveUploadResults(self, google_drive_base_folder_path)
    
    def post_run(self, *args, **kwargs) -> Any:
        self.gdrive.upload_to_google_drive()
    

class ProxyAutomation(BaseAutomationProcessFromList):
    
    def __init__(self, *args, proxies: List[str], **kwargs) -> None:
        super(ProxyAutomation, self).__init__(*args, **kwargs)
        self._proxies = proxies
    
    def start(self, *args, **kwargs):
        self.web_driver_start_args = deepcopy(args)
        self.web_driver_start_kwargs = deepcopy(kwargs)
    
    def quit(self):
        pass
    
    def get_proxy_url(self) -> str:
        return random.choice(self._proxies)
    
    def get_proxy_web_driver_handler(self) -> WebDriverHandler:
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
        return self.get_proxy_web_driver_handler()
    
    def handle_web_driver_handler_after_run(self, web_driver_handler: WebDriverHandler) -> None:
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