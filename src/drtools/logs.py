""" 
This module was created to manage logs of executions
of any .py or .ipynb file

"""

import logging
from types import FunctionType
from typing import Any, Union
import drtools.file_manager as FileManager
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


def caller_reader(f):
    """This wrapper updates the context with the callor infos"""
    def wrapper(self, *args):
        caller = getframeinfo(stack()[1][0])
        last_name = FileManager.split_path(
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
        path: str=None,
        max_bytes: int=2 * 1024 * 1024,
        backup_count: int=10,
        name: str='Main',
        default_start: bool=True,
        full_file_path_log: bool=False,
        log_as_print: bool=False
    ) -> None:
        self.full_file_path_log = full_file_path_log
        self.log_as_print = log_as_print
        
        if self.log_as_print:
            pass
        elif path is not None:
            self.LOGGER = logging.getLogger(name)
            if self.LOGGER.hasHandlers():
                self.LOGGER.handlers.clear() # Clear the handlers to avoid double logs
                
            FileManager.create_directories_of_path(path)
            logFile = RotatingFileHandler(
                path, 
                mode='a', 
                maxBytes=max_bytes,
                backupCount=backup_count, 
                encoding=None, 
                delay=0
            )
            formatter = logging.Formatter(
                '[%(threadName)-14s] [%(file)-20s] [%(asctime)s.%(msecs)03d] [%(name)-12s] [%(levelname)8s] %(message)s', 
                datefmt='%d-%m-%Y %H:%M:%S'
            )
            logFile.setFormatter(formatter)
            self.LOGGER.addHandler(logFile)
            # Here we add the Filter, think of it as a context
            self._filter = CallerFilter()
            self.LOGGER.addFilter(self._filter)
            self.LOGGER.setLevel(logging.DEBUG)        
        else:
            raise Exception('Parameter "path" must be provide or "log_as_print" must be True.')
            
        self.verbosity = True
        
        if default_start:
            self.LOGGER.info('!*************** START ***************!')

    def set_verbosity(self, verbosity: bool=True) -> None:
        """Set verbosity of logs.

        Parameters
        ----------
        verbosity : bool, optional
            If True, log all levels, 
            If False, log nothing, by default True
        """
        self.verbosity = verbosity

    def reset_verbosity(self) -> None:
        """Turn verbosity as initial state.
        """
        self.verbosity = True
            
    @caller_reader
    def debug(self, msg: any) -> None:
        """Log in DEBUG level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        if self.verbosity:
            if self.log_as_print:
                print(f'[{datetime.now().isoformat()}] [    DEBUG] {msg}')
            else:
                self.LOGGER.debug(msg)

    @caller_reader
    def info(self, msg: any) -> None:
        """Log in INFO level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        if self.verbosity:
            if self.log_as_print:
                print(f'[{datetime.now().isoformat()}] [     INFO] {msg}')
            else:
                self.LOGGER.info(msg)
        
    @caller_reader
    def warning(self, msg: any) -> None:
        """Log in WARNING level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        if self.verbosity:
            if self.log_as_print:
                print(f'[{datetime.now().isoformat()}] [  WARNING] {msg}')
            else:
                self.LOGGER.warning(msg)

    @caller_reader
    def error(self, msg: any) -> None:
        """Log in ERROR level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        if self.verbosity:
            if self.log_as_print:
                print(f'[{datetime.now().isoformat()}] [    ERROR] {msg}')
            else:
                self.LOGGER.error(msg)
        
    @caller_reader
    def critical(self, msg: any) -> None:
        """Log in CRITICAL level

        Parameters
        ----------
        msg : any
            The message that will be logged
        """
        if self.verbosity:
            if self.log_as_print:
                print(f'[{datetime.now().isoformat()}] [ CRITICAL] {msg}')
            else:
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
        # logger.debug(f'FunctionExecution : Arguments : args={args}, kwargs={kwargs}')
        response = None
        try:
            response = func(*args, **kwargs)
        except Exception as exc:
            logger.error(exc)
        logger.debug(f'FunctionExecution : End : {func.__name__}')
        return response
    return wrapper
    
    