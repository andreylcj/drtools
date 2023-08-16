""" 
This module was created to manage logs of executions
of any .py or .ipynb file

"""


import sys
import logging
from typing import Any, Union, Callable
from drtools.file_manager import (
    create_directories_of_path
)
from functools import wraps
import logging
from logging.handlers import RotatingFileHandler
from inspect import getframeinfo, stack
from datetime import datetime


class CallerFilter(logging.Filter):
    """ This class adds some context to the log record instance """
    file = ''
    line_n = ''

    def filter(self, record):
        record.file = self.file
        record.line_n = self.line_n
        return True


# def caller_reader(f):
#     """This wrapper updates the context with the callor infos"""
#     def wrapper(self, *args):
#         caller = getframeinfo(stack()[1][0])
#         last_name = split_path(
#             caller.filename
#         )[-1]
#         file = caller.filename \
#             if self.formatter_options.full_file_path_log \
#             else last_name
#         line_n = caller.lineno
#         self._filter.file = f'{file}:{line_n}'
#         return f(self, *args)
#     wrapper.__doc__ = f.__doc__
#     return wrapper


class FormatterOptions:
    def __init__(
        self,
        include_thread_name: bool=None,
        include_file_name: bool=None,
        include_datetime: bool=None,
        include_logger_name: bool=None,
        include_level_name: bool=None,
        include_exec_time: bool=None,
        exec_time_by_level: bool=None,
        full_file_path_log: bool=None,
    ) -> None:
        args = locals().copy()
        args = {k: v for k, v in args.items() if k != 'self'}
        
        start_default = True
        for k, v in args.items():
            if v is not None:
                start_default = False
                break
            
        if start_default:
            self._start_default_settings()
        else:
            for k, v in args.items():
                if v is not None:
                    setattr(self, k, v)
                else:
                    setattr(self, k, False)

    def _start_default_settings(self):
        self.include_thread_name = True
        self.include_file_name = True
        self.include_datetime = True
        self.include_logger_name = True
        self.include_level_name = True
        self.include_exec_time = False
        self.exec_time_by_level = True
        self.full_file_path_log = False
    
    @property
    def info(self):
        return {
            'include_thread_name': self.include_thread_name,
            'include_file_name': self.include_file_name,
            'include_datetime': self.include_datetime,
            'include_logger_name': self.include_logger_name,
            'include_level_name': self.include_level_name,
            'include_exec_time': self.include_exec_time,
            'exec_time_by_level': self.exec_time_by_level,
            'full_file_path_log': self.full_file_path_log,
        }


class Loggers:
    def __init__(self) -> None:
        self._loggers = {} # LoggerName -> Logger
        self._handlers = {} # LoggerName -> LogHandler
        
    def add(self, name: str, logger, handler):
        self._loggers[name] = logger
        self._handlers[name] = handler
        
    def get(self, name: str):
        return self._loggers[name]
        
    def get_or_none(self, name: str):
        return self._loggers.get(name, None)
        
    def get_handler(self, name: str):
        return self._handlers[name]
        
    def get_handler_or_none(self, name: str):
        return self._handlers.get(name, None)
    
    def delete(self, name: str):
        if name in self._loggers:
            del self._loggers[name]
        if name in self._handlers:
            del self._handlers[name]
        
        
__loggers__ = Loggers()


class Logger:
    """Handle logging
    
    Note
    -----
    You can use the max_bytes and backup_count values to allow 
    the file to rollover at a predetermined size. When the 
    size is about to be exceeded, the file is closed and 
    a new file is silently opened for output. Rollover occurs 
    whenever the current log file is nearly max_bytes in 
    length; but if either of max_bytes or backup_count is 
    zero, rollover never occurs, so you generally want 
    to set backup_count to at least 1, and have a non-zero 
    max_bytes. When backup_count is non-zero, the system 
    will save old log files by appending the 
    extensions '.1', '.2' etc., to the filename. For example, with 
    a backup_count of 5 and a base file name of app.log, you 
    would get app.log, app.log.1, app.log.2, up to app.log.5. The 
    file being written to is always app.log. When this file is 
    filled, it is closed and renamed to app.log.1, and if files 
    app.log.1, app.log.2, etc. exist, then they are renamed to 
    app.log.2, app.log.3 etc. respectively.
    
    Parameters
    ----------
    path : str
        Path to save logs
    max_bytes : int, optional
        Max bytes which one log file
        will be at maximum, by default 2*1024*1024
    backup_count : int, optional
        Number of backup logs that will be
        alive at maximum, by default 5
    name : str, optional
        Logger name, by default 'Open-Capture'
    default_start : bool, optional
        Log the initialization, by default True
    full_file_path_log : bool, optional
        If True, log file path will be complete
        If False, only will be displayed the name
        of the file, by default False
    log_level : str, optional
        Log level, by default 'DEBUG'
    formatter_options : FormatterOptions, optional
        Formatter options on logs
    """

    def __init__(
        self,
        level: int=10,
        formatter_options: FormatterOptions=FormatterOptions(),
        name: str='Main',
        path: Union[str, None]=None,
        default_start: bool=True,
        reset_logger: bool=False,
        max_bytes: int=2 * 1024 * 1024,
        backup_count: int=10,
        **kwargs
    ) -> None:
        self.level = level
        self.original_level = self.level
        self.formatter_options = formatter_options
        self.name = name
        self.path = path
        self.default_start = default_start
        self.reset_logger = reset_logger
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.started_at = datetime.now()
        self.updated_at = None
        self.updated_at_by_level = {}
        self._construct_logger()
        if self.default_start:
            self.info('!*************** START ***************!')

    def _construct_formatter(self):
        formatter_text = ''
        
        if self.formatter_options.include_datetime:
            formatter_text += '%(asctime)s.%(msecs)03d '
            
        if self.formatter_options.full_file_path_log:
            formatter_text += '{%(pathname)s:%(lineno)d} '
        elif self.formatter_options.include_file_name:
            formatter_text += '{%(filename)s:%(lineno)d} '
            
        if self.formatter_options.include_logger_name:
            formatter_text += '[%(name)-12s] '
            
        if self.formatter_options.include_thread_name:
            formatter_text += '[%(threadName)-14s] '
            
        if self.formatter_options.include_level_name:
            formatter_text += '[%(levelname)8s] '
            
        formatter_text += '%(message)s'
        
        formatter = logging.Formatter(formatter_text, datefmt='%Y-%m-%d %H:%M:%S')
        
        return formatter
    
    
    def _reset_logger(self, __loggers__):
        __loggers__.delete(self.name)
    

    def _construct_logger(self):
        global __loggers__
        
        if self.reset_logger:
            self._reset_logger(__loggers__)
        
        # Here we add the Filter, think of it as a context
        self._filter = CallerFilter()
        
        # construct formatter
        formatter = self._construct_formatter()
    
        if __loggers__.get_or_none(self.name) is not None:
            self.LOGGER = __loggers__.get(self.name)
            log_handler = __loggers__.get_handler(self.name)
            formatter = self._construct_formatter()
            log_handler.setFormatter(formatter)
            self.LOGGER.setLevel(self.level)
    
        else:
            self.LOGGER = logging.getLogger(self.name)
            
            if self.LOGGER.hasHandlers():
                self.LOGGER.handlers.clear() # Clear the handlers to avoid double logs        
                
            if self.path is not None:
                create_directories_of_path(self.path)
                log_handler = RotatingFileHandler(
                    self.path, 
                    mode='a', 
                    maxBytes=self.max_bytes,
                    backupCount=self.backup_count, 
                    encoding=None, 
                    delay=0
                )
            else:
                log_handler = logging.StreamHandler(sys.stdout)
                
            log_handler.setFormatter(formatter)
            self.LOGGER.addHandler(log_handler)
            self.LOGGER.addFilter(self._filter)
            self.LOGGER.setLevel(self.level)
                    
            __loggers__.add(self.name, self.LOGGER, log_handler)

    def set_verbosity(self, verbosity: bool=True) -> None:
        """Set verbosity of logs.

        Parameters
        ----------
        verbosity : bool, optional
            If True, log all levels, 
            If False, log nothing, by default True
        """
        if verbosity:
            self.level = 10
        else:
            self.level = 999
        self.LOGGER.setLevel(self.level)

    def reset_verbosity(self) -> None:
        """Turn verbosity as initial state.
        """
        self.level = self.original_verbosity
        self.set_level(self.level)
        
    def set_level(self, level: int):
        self.level = level
        self.LOGGER.setLevel(self.level)
        
    def _exec_seconds(self, msg_level: int=None) -> float:
        response = None
        
        now = datetime.now()
        
        previous_updated_at = self.updated_at
        self.updated_at = now
        
            
        if self.formatter_options.exec_time_by_level:
            
            if msg_level not in self.updated_at_by_level:
                self.updated_at_by_level[msg_level] = now
                response = (self.updated_at_by_level[msg_level] - self.started_at).total_seconds()
                
            else:
                response = (now - self.updated_at_by_level[msg_level]).total_seconds()
                self.updated_at_by_level[msg_level] = now
        
        else:
            
            if previous_updated_at is None:
                response = (self.updated_at - self.started_at).total_seconds()
            
            else:
                response = (self.updated_at - previous_updated_at).total_seconds()
            
        return response
    
    def _pexec_seconds(self, msg_level: int) -> str:
        return f'{round(self._exec_seconds(msg_level), 4)}s'
    
    def _insert_exec_time_on_message(self, msg: str, msg_level: int) -> str:
        exec_time = self._pexec_seconds(msg_level).rjust(10, " ")
        msg = f'[{exec_time}] {msg}'
        return msg
    
    # @caller_reader
    def debug(self, msg: any) -> None:
        """Log in DEBUG level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        if self.formatter_options.include_exec_time:
            msg = self._insert_exec_time_on_message(msg, 10)
        self.LOGGER.debug(msg)

    # @caller_reader
    def info(self, msg: any) -> None:
        """Log in INFO level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        if self.formatter_options.include_exec_time:
            msg = self._insert_exec_time_on_message(msg, 20)
        self.LOGGER.info(msg)
        
    # @caller_reader
    def warning(self, msg: any) -> None:
        """Log in WARNING level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        if self.formatter_options.include_exec_time:
            msg = self._insert_exec_time_on_message(msg, 30)
        self.LOGGER.warning(msg)

    # @caller_reader
    def error(self, msg: any) -> None:
        """Log in ERROR level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        if self.formatter_options.include_exec_time:
            msg = self._insert_exec_time_on_message(msg, 40)
        self.LOGGER.error(msg)
        
    # @caller_reader
    def critical(self, msg: any) -> None:
        """Log in CRITICAL level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        if self.formatter_options.include_exec_time:
            msg = self._insert_exec_time_on_message(msg, 50)
        self.LOGGER.critical(msg)
    

def function_name_start_and_end(
    func: Callable,
    logger: Logger
) -> Callable:
    """Log name of function.
    
    Logs the name of function on start and end of execution.
    Logs error too.

    Parameters
    ----------
    func : FunctionType
        Function that will be executed
    logger : Logger, optional
        Specific logger, by default logging

    Returns
    -------
    FunctionType
        The wrapper function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Union[Any, None]:
        logger.debug(f'FunctionExecution : Start : {func.__name__}()')
        response = None
        try:
            response = func(*args, **kwargs)
        except Exception as exc:
            logger.error(exc)
        logger.debug(f'FunctionExecution : End : {func.__name__}')
        return response
    return wrapper