"""This module was created to handle concurrent executions

"""

import concurrent
import concurrent.futures
from datetime import datetime
from typing import List, Callable
from drtools.utils import progress, display_time
from drtools.logging import Logger, FormatterOptions
import math
import traceback


WorkerData = any
"""any: Worker data can be any type of data."""


class Worker:
    def __init__(
        self,
        parameters: any,
        num: int,
        verbosity: bool=False,
    ) -> None:
        self.parameters = parameters
        self.num = num
        self.verbosity = verbosity


class ThreadPoolExecutor:
    
    def __init__(
        self,
        exec_func: Callable,
        worker_data: List[WorkerData],
        max_workers: int=5,
        verbose: int=100,
        verbose_parameters_sample: bool=False,
        direct: bool=True,
        log_traceback: bool=False,
        LOGGER: Logger=Logger(
            name='ThreadPoolExecutorMain',
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_logger_name=True,
                include_level_name=True,
            ),
            default_start=False
        )
    ) -> None:
        self.exec_func = exec_func
        self.worker_data = worker_data
        self.max_workers = max_workers
        self.verbose = verbose
        self.verbose_parameters_sample = verbose_parameters_sample
        self.direct = direct
        self.log_traceback = log_traceback
        self.LOGGER = LOGGER
        self.total_worker_data_len = len(self.worker_data)
        
    def start(self) -> None:
        """Start Thread Pool Execution."""
        
        self.num_of_processed_workers = 0  
        self.started_at = datetime.now()
        
        if self.total_worker_data_len == 0:
            self.LOGGER.info('No data to process.')
            return None
        
        self.LOGGER.info('Starting Thread Pool Execution...')
        self._start() 
        self.LOGGER.info('Thread Pool Execution Finished.')
        
        
    def _start(self) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._track_progress,
                    Worker(
                        parameters=worker, 
                        num=worker_num+1, 
                        verbosity=worker_num % self.verbose == 0 if self.verbose > 0 \
                            else False
                    )
                ): worker
                for worker_num, worker in enumerate(self.worker_data)
            }
            for future in concurrent.futures.as_completed(futures):
                worker = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    self.LOGGER.error(f'Worker {worker} generated an exception: {exc}')
                    if self.log_traceback:
                        self.LOGGER.error(traceback.format_exc())
                
    
    def _track_progress(self, worker: Worker) -> any:
        exec_func_response = self._exec_func_handler(worker)
        self.num_of_processed_workers = self.num_of_processed_workers + 1
        
        if worker.verbosity:
            total_exec_time = (datetime.now() - self.started_at).total_seconds()
            progress_percentage = progress(
                current=self.num_of_processed_workers, 
                total=self.total_worker_data_len
            )
            seconds_by_worker = total_exec_time / self.num_of_processed_workers
            expected_remaining_seconds = math.ceil(
                (self.total_worker_data_len - self.num_of_processed_workers) * seconds_by_worker
            )
            expected_remaining_seconds = expected_remaining_seconds + 1
            
            log_txt = f'{progress_percentage}% ({self.num_of_processed_workers:,}/{self.total_worker_data_len:,}) complete. '
            log_txt += f'Expected remaining time: {display_time(expected_remaining_seconds)}'
            self.LOGGER.info(log_txt)
            
    
    def _exec_func_handler(self, worker: Worker):
        started_at = datetime.now()
        
        parameters_str = str(worker.parameters)
        parameters_sample = f'{parameters_str[:100]} ... {parameters_str[-100:]}'
        
        if worker.verbosity:
            if self.verbose_parameters_sample:
                self.LOGGER.info(f'Start execution with parameters (sample): {parameters_sample}')
            else:
                self.LOGGER.info(f'Start execution with parameters: {worker.parameters}')
                
        func_response = None
        
        try:
            func_parameters = worker.parameters if self.direct else worker
            func_response = self.exec_func(func_parameters)
            
            if worker.verbosity:
                log_text = f'Succesful execution! Execution response: '
                response_str = str(func_response)
                log_text += f'{response_str[:100]}'
                
                if len(response_str) > 100:
                    log_text += f' <|:::|> {response_str[-min(len(response_str) - 100, 100):]}'
                
                self.LOGGER.info(log_text)
        
        except Exception as exc:
            if self.verbose_parameters_sample:
                self.LOGGER.error(f'Execution with parameters (sample): {parameters_sample} generate an exception: {exc}')
            else:
                self.LOGGER.error(f'Execution with parameters: {worker.parameters} generate an exception: {exc}')
                
            if self.log_traceback:
                self.LOGGER.error(traceback.format_exc())
                
        finally:
            if worker.verbosity:
                time_diff = (datetime.now() - started_at).total_seconds()
                self.LOGGER.info(f"Execution ends in {time_diff}s.")