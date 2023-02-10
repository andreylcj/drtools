""" 
This module was created to manage logs of executions
of any .py or .ipynb file

"""


import sys
import logging
from types import FunctionType
from typing import Any, Union, TypedDict
from drtools.file_manager import (
    split_path, create_directories_of_path
)
from functools import wraps
import logging
from logging.handlers import RotatingFileHandler
from inspect import getframeinfo, stack


class CallerFilter(logging.Filter):
    """ This class adds some context to the log record instance """
    file = ''
    line_n = ''

    def filter(self, record):
        record.file = self.file
        record.line_n = self.line_n
        return True


def caller_reader(f):
    """This wrapper updates the context with the callor infos"""
    def wrapper(self, *args):
        caller = getframeinfo(stack()[1][0])
        last_name = split_path(
            caller.filename
        )[-1]
        file = caller.filename \
            if self.full_file_path_log \
            else last_name
        line_n = caller.lineno
        self._filter.file = f'{file}:{line_n}'
        return f(self, *args)
    wrapper.__doc__ = f.__doc__
    return wrapper


class FormatterOptions(TypedDict):
    IncludeThreadName: bool
    IncludeFileName: bool
    IncludeDate: bool
    IncludeLoggerName: bool
    IncludeLevelName: bool


class Log:
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
    """

    def __init__(
        self,
        path: Union[str, None]=None,
        max_bytes: int=2 * 1024 * 1024,
        backup_count: int=10,
        name: str='Main',
        default_start: bool=True,
        full_file_path_log: bool=False,
        log_level: str='DEBUG',
        formatter_options: FormatterOptions=FormatterOptions({
            'IncludeThreadName': True,
            'IncludeFileName': True,
            'IncludeDate': True,
            'IncludeLoggerName': True,
            'IncludeLevelName': True,
        }),
        # deprecated 
        log_as_print: bool=False
    ) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.name = name
        self.default_start = default_start
        self.full_file_path_log = full_file_path_log
        self.log_level = log_level
        self.formatter_options = formatter_options
        self.log_level = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}[self.log_level]        
        self.original_verbosity = self.log_level
        self._construct_logger()
        if default_start:
            self.info('!*************** START ***************!')

    def _construct_logger(
        self,
    ):
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
        formatter_text = ''
        accept_conditions = [True]
        if self.formatter_options.get('IncludeThreadName', None) in accept_conditions:
            formatter_text = '[%(threadName)-14s] '
        if self.formatter_options.get('IncludeFileName', None) in accept_conditions:
            formatter_text = formatter_text + '[%(file)-20s] '
        if self.formatter_options.get('IncludeDate', None) in accept_conditions:
            formatter_text = formatter_text + '[%(asctime)s.%(msecs)03d] '
        if self.formatter_options.get('IncludeLoggerName', None) in accept_conditions:
            formatter_text = formatter_text + '[%(name)-12s] '
        if self.formatter_options.get('IncludeLevelName', None) in accept_conditions:
            formatter_text = formatter_text + '[%(levelname)8s] '        
        formatter = logging.Formatter(formatter_text + '%(message)s', datefmt='%d-%m-%Y %H:%M:%S')
        log_handler.setFormatter(formatter)
        self.LOGGER.addHandler(log_handler)
        # Here we add the Filter, think of it as a context
        self._filter = CallerFilter()
        self.LOGGER.addFilter(self._filter)
        self.LOGGER.setLevel(self.log_level)

    def set_verbosity(self, verbosity: bool=True) -> None:
        """Set verbosity of logs.

        Parameters
        ----------
        verbosity : bool, optional
            If True, log all levels, 
            If False, log nothing, by default True
        """
        if verbosity:
            self.log_level = 10
        else:
            self.log_level = 999
        self.LOGGER.setLevel(self.log_level)

    def reset_verbosity(self) -> None:
        """Turn verbosity as initial state.
        """
        self.log_level = self.original_verbosity
        self.LOGGER.setLevel(self.log_level)
    
    @caller_reader
    def debug(self, msg: any) -> None:
        """Log in DEBUG level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        self.LOGGER.debug(msg)

    @caller_reader
    def info(self, msg: any) -> None:
        """Log in INFO level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        self.LOGGER.info(msg)
        
    @caller_reader
    def warning(self, msg: any) -> None:
        """Log in WARNING level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        self.LOGGER.warning(msg)

    @caller_reader
    def error(self, msg: any) -> None:
        """Log in ERROR level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        self.LOGGER.error(msg)
        
    @caller_reader
    def critical(self, msg: any) -> None:
        """Log in CRITICAL level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        self.LOGGER.critical(msg)
    

def function_name_start_and_end(
    func: FunctionType,
    logger: Log
) -> FunctionType:
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