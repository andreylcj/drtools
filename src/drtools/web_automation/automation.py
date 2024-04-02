

from .driver_handler.handler import WebDriverHandler
from .driver_handler.chrome import ChromeWebDriverHandler
from typing import List, Any, Callable, Union
from drtools.logging import Logger, FormatterOptions
import uuid
from datetime import datetime
import time
from drtools.utils import display_time
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


class BaseAutomationProcess:
    
    WEB_DRIVER_HANDLER_CLASS: Union[ChromeWebDriverHandler]=ChromeWebDriverHandler
    
    def __init__(
        self, 
        driver: WebDriver=None,
        LOGGER: Logger=None,
        start: bool=False,
        quit: bool=False,
    ) -> None:
        self.web_driver_handler = None
        if not LOGGER:
            LOGGER = Logger(
                name="BaseAutomationProcess",
                formatter_options=FormatterOptions(include_datetime=True, include_logger_name=True, include_level_name=True),
                default_start=False
            )
        self.LOGGER = LOGGER
        self._start = start
        self._quit = quit
        if driver:
            self.web_driver_handler = self.WEB_DRIVER_HANDLER_CLASS(driver, LOGGER)
        self._result = None
        
    def start(self, *args, **kwargs):
        self.LOGGER.info(f"Initializing driver...")
        web_driver_handler = self.WEB_DRIVER_HANDLER_CLASS()
        web_driver_handler.start(*args, **kwargs)
        self.web_driver_handler = web_driver_handler
        self.LOGGER.info("Initializing driver... Done!")
    
    def quit(self):
        self.web_driver_handler.quit()
    
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
            self.set_execution_id(str(uuid.uuid4()))
            automation_result: Any = self.run_executor(*args, **kwargs)
            self.set_result(
                AutomationResult(
                    execution_id=self.get_execution_id(),
                    started_at=str(started_at),
                    finished_at=str(datetime.now()),
                    result=automation_result,
                )
            )
    
    def run_executor(self, *args, **kwargs) -> Any:
        return self.run(self.web_driver_handler, *args, **kwargs)
    
    def run(self, web_driver_handler: WebDriverHandler, *args, **kwargs) -> Any:
        raise NotImplementedError


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
        wait_pre_action: Callable=None,
        wait_post_action: Callable=None,
        max_workers: int=1,
        proxies: List[str]=[],
    ) -> None:
        super(BaseAutomationProcessFromList, self).__init__(None, LOGGER, start, quit)
        self.raise_exception = raise_exception
        self.bulk_size = bulk_size
        self.wait_time = wait_time
        self.wait_pre_action = wait_pre_action
        self.wait_post_action = wait_post_action
        self.max_workers = max_workers
        self.proxies = proxies
        self._web_driver_handlers = []
        self._lock = Lock()
        self._success_executions_by_handler = {} # web_driver_handler -> execution_count
        self._errors_executions_by_handler = {} # web_driver_handler -> execution_count
        self.web_driver_start_args = ()
        self.web_driver_start_kwargs = {}
        
    def set_web_driver_start_args_kwargs(self, *args, **kwargs):
        self.web_driver_start_args = args
        kwargs['LOGGER'] = self.LOGGER
        self.web_driver_start_kwargs = kwargs
        
    def start(self, *args, **kwargs):
        self.LOGGER.info(f"Initializing drivers...")
        for i in range(self.max_workers):
            web_driver_handler = self.WEB_DRIVER_HANDLER_CLASS(LOGGER=self.LOGGER)
            web_driver_handler.start(*args, **kwargs)
            self._web_driver_handlers.append(web_driver_handler)
            self._success_executions_by_handler[web_driver_handler] = 0
            self._errors_executions_by_handler[web_driver_handler] = 0
        self.LOGGER.info("Initializing drivers... Done!")
    
    def quit(self):
        self.LOGGER.info(f"Quiting drivers...")
        for web_driver_handler in self._web_driver_handlers:
            web_driver_handler.quit()
        self.LOGGER.info(f"Quiting drivers Done!...")
    
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
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.run_middleware,
                    Worker(list_item=list_item, list_item_idx=list_item_idx, started_at=started_at, total=total, args=args, kwargs=kwargs)
                ): list_item
                for list_item_idx, list_item in enumerate(list_items)
            }
            for future in futures_as_completed(futures):
                future.result()
    
    def run_middleware(self, worker: Worker) -> None:
        list_item = worker['list_item']
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
            list_item_result: Any = self.run(web_driver_handler, list_item, list_item_idx, *args, **kwargs)
            self.increment_automation_success_count()
            self.increment_web_driver_handler_success_count(web_driver_handler)
        except Exception as exc:
            if self.raise_exception:
                raise exc
            error = str(exc)
            error_traceback = traceback.format_exc()
            self.LOGGER.error(error)
            if self.verbose_traceback:
                self.LOGGER.error(error_traceback)
            self.increment_automation_error_count()
            self.increment_web_driver_handler_error_count(web_driver_handler)
        self.handle_web_driver_handler_after_run(web_driver_handler)
        self.append_automation_result(
                AutomationFromListItemResult(
                started_at=str(item_started_at),
                finished_at=str(datetime.now()),
                error=error,
                error_traceback=error_traceback,
                list_item_result=list_item_result,
                list_item=list_item,
            )
        )
        processed_items_num = self.get_automation_processed_items_num()
        total_time = (datetime.now()-started_at).total_seconds()
        speed = total_time / processed_items_num
        remaining_time = (total - processed_items_num) * speed
        remaining_time_msg = display_time(int(remaining_time))
        self.LOGGER.debug(f"({processed_items_num:,}/{total:,}) Complete! Expected remaining time: {remaining_time_msg}...")
        if self.wait_time \
        and not self.proxies:
            if not self.bulk_size:
                raise Exception("When wait_time is not None, bulk_size must be set.")
            processed_items_num = self.get_web_driver_handler_execution_count(web_driver_handler)
            if processed_items_num % self.bulk_size == 0:
                if self.wait_pre_action:
                    self.LOGGER.debug('Waiting pre action...')
                    self.wait_pre_action()
                    self.LOGGER.debug('Waiting pre action... Done!')
                self.LOGGER.debug(f'Waiting for {self.wait_time:,}s...')
                time.sleep(self.wait_time)
                self.LOGGER.debug(f'Waiting for {self.wait_time:,}s... Done!')
                if self.wait_post_action:
                    self.LOGGER.debug('Waiting post action...')
                    self.wait_post_action()
                    self.LOGGER.debug('Waiting post action... Done!')
    
    #########################
    
    def get_proxy_web_driver_handler(self) -> WebDriverHandler:
        random_proxy_server = random.choice(self.proxies)
        seleniumwire_options = {
            'proxy': {
                'http': f'{random_proxy_server}',
                'https': f'{random_proxy_server}',
                'verify_ssl': False,
            },
        }
        original_seleniumwire_options = self.web_driver_start_kwargs.pop('seleniumwire_options', {})
        seleniumwire_options = {**original_seleniumwire_options, **seleniumwire_options}
        web_driver_handler: WebDriverHandler = self.WEB_DRIVER_HANDLER_CLASS(LOGGER=self.LOGGER)
        web_driver_handler.start(
            *self.web_driver_start_args, 
            seleniumwire_options=seleniumwire_options,
            **self.web_driver_start_kwargs
        )
        return web_driver_handler
    
    def pop_web_driver_handler(self) -> WebDriverHandler:
        if self.proxies:
            return self.get_proxy_web_driver_handler()
        else:
            with self._lock:
                web_driver_handler: WebDriverHandler = self._web_driver_handlers.pop(0)
                return web_driver_handler
    
    def handle_web_driver_handler_after_run(self, web_driver_handler: WebDriverHandler) -> None:
        if self.proxies:
            web_driver_handler.quit()
        else:
            with self._lock:
                self._web_driver_handlers.append(web_driver_handler)
    
    def get_web_driver_handlers(self) -> List[WebDriverHandler]:
        return self._web_driver_handlers
    
    def get_web_driver_handler_execution_count(self, web_driver_handler: WebDriverHandler) -> int:
        return self._success_executions_by_handler[web_driver_handler] + self._errors_executions_by_handler[web_driver_handler]
        
    def increment_web_driver_success_execution_count(self, web_driver_handler: WebDriverHandler) -> None:
        if not self.proxies:
            with self._lock:
                self._success_executions_by_handler[web_driver_handler] += 1
            
    def increment_web_driver_handler_error_count(self, web_driver_handler: WebDriverHandler) -> None:
        if not self.proxies:
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
    
    def run(self, web_driver_handler: WebDriverHandler, list_item: Any, list_item_idx: int, *args, **kwargs) -> Any:
        raise NotImplementedError


# class ThreadingAutomationFromList(BaseAutomationProcessFromList):
    
#     def __init__(self, *args, **kwargs) -> None:
#         super(ThreadingAutomationFromList, self).__init__(*args, **kwargs)
#         self._web_driver_handlers = []
#         self._lock = Lock()
#         self._executions_by_handler = {} # web_driver_handler -> execution_count
        
#     def initialize_driver(
#         self, 
#         *args, 
#         web_driver_handler_class: Union[ChromeWebDriverHandler]=ChromeWebDriverHandler,
#         max_workers: int=4, 
#         **kwargs
#     ):
#         self.LOGGER.info(f"Initializing drivers...")
#         self.max_workers = max_workers
#         for i in range(max_workers):
#             web_driver_handler = web_driver_handler_class(LOGGER=self.LOGGER)
#             web_driver_handler.start(*args, **kwargs)
#             self._web_driver_handlers.append(web_driver_handler)
#             self._executions_by_handler[web_driver_handler] = 0
#         self.LOGGER.info("Initializing drivers... Done!")
    
#     def quit_driver(self):
#         self.LOGGER.info(f"Quiting drivers...")
#         for web_driver_handler in self._web_driver_handlers:
#             web_driver_handler.quit()
#         self.LOGGER.info(f"Quiting drivers Done!...")
    
#     def process_list_items(
#         self,
#         list_items: List[Any],
#         started_at: datetime,
#         total: int,
#         raise_exception: bool=False,
#         bulk_size: int=None,
#         wait_time: int=None,
#         wait_pre_action: Callable=None,
#         wait_post_action: Callable=None,
#         verbose_traceback: bool=False,
#         **kwargs,
#     ):
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = {
#                 executor.submit(
#                     self.run_single_executor_middleware,
#                     Worker(
#                         list_item=list_item, 
#                         list_item_idx=list_item_idx,
#                         started_at=started_at,
#                         total=total,
#                         raise_exception=raise_exception,
#                         bulk_size=bulk_size,
#                         wait_time=wait_time,
#                         wait_pre_action=wait_pre_action,
#                         wait_post_action=wait_post_action,
#                         verbose_traceback=verbose_traceback,
#                         kwargs=kwargs
#                     )
#                 ): list_item
#                 for list_item_idx, list_item in enumerate(list_items)
#             }
#             for future in futures_as_completed(futures):
#                 future.result()
    
#     def run_single_executor_middleware(self, worker: Worker) -> Any:
#         web_driver_handler = self.get_web_driver_handler()
#         web_driver_handler_execution_count = self.get_web_driver_handler_execution_count(web_driver_handler)
#         run_response = super(ThreadingAutomationFromList, self).run_single_executor_middleware(
#             worker['list_item'],
#             worker['list_item_idx'],
#             worker['started_at'],
#             worker['total'],
#             worker['raise_exception'],
#             worker['bulk_size'],
#             worker['wait_time'],
#             worker['wait_pre_action'],
#             worker['wait_post_action'],
#             worker['verbose_traceback'],
#             web_driver_handler,
#             web_driver_handler_execution_count,
#             **worker['kwargs'],
#         )
#         self.increment_web_driver_handler_execution_count(web_driver_handler)
#         self.add_web_driver_handler(web_driver_handler)
#         return run_response
    
#     def run_single_executor(
#         self, 
#         web_driver_handler: WebDriverHandler,
#         web_driver_handler_execution_count: int, 
#         list_item: Any, 
#         list_item_idx: int, 
#         **kwargs
#     ) -> Any:
#         return self.run(
#             web_driver_handler,
#             web_driver_handler_execution_count, 
#             list_item, 
#             list_item_idx, 
#             **kwargs
#         )
    
#     def run(
#         self, 
#         web_driver_handler: WebDriverHandler,
#         web_driver_handler_execution_count: int, 
#         list_item: Any, 
#         list_item_idx: int, 
#         **kwargs
#     ) -> Any:
#         raise NotImplementedError
    
#     #########################
    
#     def get_web_driver_handler(self) -> WebDriverHandler:
#         with self._lock:
#             web_driver_handler: WebDriverHandler = self._web_driver_handlers.pop(0)
#             return web_driver_handler
    
#     def get_web_driver_handlers(self) -> List[WebDriverHandler]:
#         return self._web_driver_handlers
    
#     def add_web_driver_handler(self, web_driver_handler: WebDriverHandler) -> None:
#         with self._lock:
#             self._web_driver_handlers.append(web_driver_handler)
    
#     def get_web_driver_handler_execution_count(self, web_driver_handler: WebDriverHandler) -> int:
#         return self._executions_by_handler[web_driver_handler]
        
#     def increment_web_driver_handler_execution_count(self, web_driver_handler: WebDriverHandler) -> None:
#         with self._lock:
#             self._executions_by_handler[web_driver_handler] += 1
    
#     #########################
    
#     def increment_automation_success_count(self) -> None:
#         with self._lock:
#             return super(ThreadingAutomationFromList, self).increment_automation_success_count()
        
#     def increment_automation_error_count(self) -> None:
#         with self._lock:
#             return super(ThreadingAutomationFromList, self).increment_automation_error_count()
    
#     def append_automation_result(self, automation_result: AutomationFromListItemResult) -> None:
#         with self._lock:
#             return super(ThreadingAutomationFromList, self).append_automation_result(automation_result)


# class ThreadingGetEcommerceProductFromProductUrlList(ThreadingAutomationFromList):
        
#     def run_single_executor(
#         self, 
#         web_driver_handler: WebDriverHandler,
#         web_driver_handler_execution_count: int, 
#         product_url: EcommerceProductUrl,
#         list_item_idx: int, 
#         **kwargs
#     ) -> EcommerceProduct:
#         ecommerce_product: EcommerceProduct = EcommerceProduct()
#         ecommerce_product['datetime'] = str(datetime.now())
#         ecommerce_product['url_info'] = get_url_info(product_url)
#         run_response = self.run(
#             web_driver_handler, 
#             web_driver_handler_execution_count,
#             product_url, 
#             list_item_idx, 
#             **kwargs
#         )
#         ecommerce_product: EcommerceProduct = {**ecommerce_product, **run_response}
#         return ecommerce_product
        
#     def run(
#         self, 
#         web_driver_handler: WebDriverHandler,
#         web_driver_handler_execution_count: int, 
#         product_url: EcommerceProductUrl,
#         list_item_idx: int, 
#         **kwargs
#     ) -> EcommerceProduct:
#         raise NotImplementedError
    

# class ThreadingGenericGetEcommerceProductFromProductUrlList(ThreadingGetEcommerceProductFromProductUrlList):
    
#     def run(
#         self, 
#         web_driver_handler: WebDriverHandler,
#         web_driver_handler_execution_count: int, 
#         product_url: EcommerceProductUrl, 
#         list_item_idx: int, 
#         price_restrictions: PriceRestrictions=None,
#         top_prices_num: int=99999,
#         exclude_tags_and_childs: List[str]=DEFAULT_EXCLUDE_TAGS_AND_CHILDS,
#         **kwargs
#     ) -> EcommerceProduct:
#         return GenericGetEcommerceProductFromProductUrlList.get_possible_prices(
#             web_driver_handler,
#             product_url,
#             web_driver_handler_execution_count,
#             price_restrictions,
#             top_prices_num,
#             exclude_tags_and_childs,
#             **kwargs
#         )