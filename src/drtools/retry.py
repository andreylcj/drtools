

from typing import Callable, Any, Dict, List
from datetime import datetime
import traceback
import time
from drtools.logging import FormatterOptions, Logger
from enum import Enum
import pytz


class RetryType(Enum):
    EXPONENCIAL = 'Exponencial'
    STATIC = 'Static'


def compute_retry_wait_time(attempts: int, retry_wait_time: int, retry_type: RetryType) -> int:
    if retry_type is RetryType.EXPONENCIAL:
        return retry_wait_time ** (attempts-1)
    if retry_type is RetryType.STATIC:
        return retry_wait_time
    raise Exception(f'Retry type {retry_type} not supported.')


class CustomKwargs:
    def __init__(self) -> None:
        self._items = {}
    
    def add(self, name, value):
        self._items[name] = value
    
    def get(self, name):
        return self._items[name]
    
    def get_or(self, name, or_value):
        return self._items.get(name, or_value)
    
    def get_or_none(self, name):
        return self.get_or(name, None)


def try_statement(*args, custom_kwargs: CustomKwargs, attempts: int, **kwargs):
    pass
    
def except_statement(*args, custom_kwargs: CustomKwargs, attempts: int, exception, **kwargs):
    pass
    
def finally_statement(*args, custom_kwargs: CustomKwargs, attempts: int, exception, **kwargs):
    pass


class RetryConfig:
    def __init__(
        self,       
        name: str='RetryMain',
        max_retry_attempts: int=50,
        retry_wait_time: int=10,
        retry_type: RetryType=RetryType.STATIC,
        log_includes_traceback: bool=False,
        LOGGER: Logger=Logger(
            name='RetryHandler',
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_thread_name=True,
                include_logger_name=True,
            ),
            default_start=False
        )
    ) -> None:
        self.name = name
        self.max_retry_attempts = max_retry_attempts
        self.retry_wait_time = retry_wait_time
        self.retry_type = retry_type
        self.log_includes_traceback = log_includes_traceback
        self.except_statement = except_statement
        self.LOGGER = LOGGER
        
        
class StatementType(Enum):
    TRY = 'TRY'
    EXCEPT = 'EXCEPT'
    FAIL_FINALLY_BEFORE_WAIT_TIME = 'FAIL_FINALLY_BEFORE_WAIT_TIME'
    FAIL_FINALLY_AFTER_WAIT_TIME = 'FAIL_FINALLY_AFTER_WAIT_TIME'
    SUCCESS_FINALLY = 'SUCCESS_FINALLY'


class StatementInfo:
    def __init__(
        self,       
        info: Any,
        statement: StatementType,
    ):
        self._info = info
        self.statement = statement
        self.created_at = datetime.now().replace(tzinfo=pytz.UTC)
        
    @property
    def info(self) -> Dict: 
        return {
            'info': self._info,
            'statement': self.statement.value,
            'created_at': str(self.created_at)
        }
        

class StatementInfoArchive:
    def __init__(self) -> None:
        self._infos: Dict[StatementType, List[StatementInfo]] = {}
    
    def add_info(self, statement_info: StatementInfo):
        if statement_info.statement not in self._infos:
            self._infos[statement_info.statement] = []
        self._infos[statement_info.statement].append(statement_info)
        
    @property
    def info(self) -> Dict: 
        return {
            statement.value: [
                statement_info.info
                for statement_info in statement_info_list
            ]
            for statement, statement_info_list in self._infos.items()
        }


class RetryHandler:
    def __init__(
        self,
        try_statement: Callable,
        retry_config: RetryConfig,
        except_statement: Callable=except_statement,
        fail_finally_statement_before_wait_time: Callable=finally_statement,
        fail_finally_statement_after_wait_time: Callable=finally_statement,
        success_finally_statement: Callable=finally_statement,
        raise_exception_handler: Callable=None,
        log_error_extra_information: Any=None,
    ) -> None:
        self.try_statement = try_statement
        self.name = retry_config.name
        self.max_retry_attempts = retry_config.max_retry_attempts
        self.retry_wait_time = retry_config.retry_wait_time
        self.retry_type = retry_config.retry_type
        self.log_includes_traceback = retry_config.log_includes_traceback
        self.except_statement = except_statement
        self.success_finally_statement = success_finally_statement
        self.fail_finally_statement_before_wait_time = fail_finally_statement_before_wait_time
        self.fail_finally_statement_after_wait_time = fail_finally_statement_after_wait_time
        if raise_exception_handler is None:
            def raise_exception_handler_(attempts):
                raise Exception(f'No success execution after reach total retries attempts #{attempts:,}.')
            self.raise_exception_handler = raise_exception_handler_
        else:
            self.raise_exception_handler = raise_exception_handler
        self.log_error_extra_information = log_error_extra_information
        self.LOGGER = retry_config.LOGGER
        self.custom_kwargs = CustomKwargs()
        self.statement_info_archive = StatementInfoArchive()
    
    @property
    def retry_name(self) -> str:
        return self.name.rjust(10, " ")
    
    def _prepare_error_txt(
        self,
        attempts: int, 
        duration: float, 
        sleep_for: float, 
        exception_txt: str, 
        traceback_txt: str=None,
        extra_information: Any=None,
    ):
        log_txt = "[{}] [Attempt #{}/#{}] Error"
        format_args = [
            self.retry_name, 
            attempts, 
            self.max_retry_attempts
        ]
            
        log_txt += " after {}s: {} - sleeping for {}s and will retry."
        format_args = format_args + [duration, exception_txt, sleep_for]
        
        if self.log_includes_traceback:
            log_txt = log_txt + "\n{}"
            format_args.append(traceback_txt)
        
        if extra_information:
            if self.log_includes_traceback:
                log_txt += "\n\n"
            else:
                log_txt += " "
            log_txt += "Extra Information: {}"
            if self.log_includes_traceback:
                log_txt += "\n"
            format_args.append(extra_information)
        
        format_args = tuple(format_args)
        
        return log_txt.format(*format_args)
    
    def _prepare_attempts_txt(
        self, 
        attempts: int, 
        extra_information: Any=None,
    ):
        log_txt = f"[{self.retry_name}] Success after #{attempts:,} from a total of #{self.max_retry_attempts} attempts."
        if extra_information:
            log_txt += f" Extra Information: {extra_information}"
        return log_txt
    
    def run(self, *args, **kwargs):
        
        # process_started_at = datetime.now()
        requesting_process_failed = True
        attempts = 0
        
        while attempts < self.max_retry_attempts \
        and requesting_process_failed:
            
            attempts += 1
            requesting_process_failed = True
            
            try:
                started_at = datetime.now()
                try_statement_response = self.try_statement(
                    *args, 
                    custom_kwargs=self.custom_kwargs, 
                    attempts=attempts, 
                    **kwargs
                )
                self.statement_info_archive.add_info(
                    StatementInfo(
                        info=try_statement_response,
                        statement=StatementType.TRY
                    )
                )
                requesting_process_failed = False
            
            except Exception as exc:
                requesting_process_failed = True
                exception = exc
                exception_txt = str(exc).rstrip().lstrip()
                traceback_txt = traceback.format_exc().rstrip().lstrip()
                except_statement_response = self.except_statement(
                    *args, 
                    custom_kwargs=self.custom_kwargs, 
                    attempts=attempts, 
                    exception=exception, 
                    **kwargs
                )
                self.statement_info_archive.add_info(
                    StatementInfo(
                        info=except_statement_response,
                        statement=StatementType.EXCEPT
                    )
                )
            
            finally:
                if requesting_process_failed:
                    duration = round((datetime.now() - started_at).total_seconds(), 4)
                    sleep_for = compute_retry_wait_time(
                        attempts=attempts, 
                        retry_wait_time=self.retry_wait_time, 
                        retry_type=self.retry_type
                    )
                    log_txt = self._prepare_error_txt(
                        attempts=attempts,
                        duration=duration,
                        exception_txt=exception_txt,
                        sleep_for=sleep_for,
                        traceback_txt=traceback_txt,
                        extra_information=self.log_error_extra_information,
                    )
                    self.LOGGER.error(log_txt)
                    
                
                if requesting_process_failed:
                    fail_finally_statement_before_wait_time_response = self.fail_finally_statement_before_wait_time(
                        *args, 
                        custom_kwargs=self.custom_kwargs, 
                        attempts=attempts, 
                        exception=exception, 
                        **kwargs
                    )
                    self.statement_info_archive.add_info(
                        StatementInfo(
                            info=fail_finally_statement_before_wait_time_response,
                            statement=StatementType.FAIL_FINALLY_BEFORE_WAIT_TIME
                        )
                    )
                    
                    # wait for retry
                    time.sleep(sleep_for)
                    
                    fail_finally_statement_after_wait_time_response = self.fail_finally_statement_after_wait_time(
                        *args, 
                        custom_kwargs=self.custom_kwargs, 
                        attempts=attempts, 
                        exception=exception, 
                        **kwargs
                    )
                    self.statement_info_archive.add_info(
                        StatementInfo(
                            info=fail_finally_statement_after_wait_time_response,
                            statement=StatementType.FAIL_FINALLY_AFTER_WAIT_TIME
                        )
                    )
                    
                else:
                    success_finally_statement_response = self.success_finally_statement(
                        *args, 
                        custom_kwargs=self.custom_kwargs, 
                        attempts=attempts, 
                        exception=None, 
                        **kwargs
                    )
                    self.statement_info_archive.add_info(
                        StatementInfo(
                            info=success_finally_statement_response,
                            statement=StatementType.SUCCESS_FINALLY
                        )
                    )
        
        if attempts > 1 and not requesting_process_failed:
            log_txt = self._prepare_attempts_txt(
                attempts=attempts,
                extra_information=self.log_error_extra_information,
            )
            self.LOGGER.info(log_txt)
        
        if requesting_process_failed:
            self.raise_exception_handler(attempts)
            
    def get_statement_info_archive(self) -> StatementInfoArchive:
        return self.statement_info_archive
        