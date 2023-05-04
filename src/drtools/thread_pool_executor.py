"""This module was created to handle concurrent executions

"""


import concurrent
import concurrent.futures
import logging
from datetime import datetime
from typing import Dict, List
from types import FunctionType
from drtools.utils import progress, display_time
from drtools.logs import Log
import numpy as np
from numpy import inf
import math


WorkerData = any
"""any: Worker data can be any type of data."""

LambdaResponse = any
"""any: Lambda Response can be any type of data."""

class Event:
	"""Lambda event to pass to ``handle_lambda`` function

	Parameters
	----------
	parameters : any
		Parameters to pass to ``execution_function``
	execution_function : FunctionType
		The function that will be executed
	verbose : bool, optional
		If lambda execution will have logs, by default True.
	verbose_parameters_sample : bool, optional
		If True, will log only a sample of parameters, 
  		by default True.
	direct : bool, optional
		If True, pass parameters direct to execution function 
  		as arguments, by default False.
	LOGGER : Log, optional
		The Logger, by default None
	
	"""
    
	def __init__(
		self,
		parameters: any,
		execution_function: FunctionType,
		verbose: bool=True,
  		verbose_parameters_sample: bool=True,
		direct: bool=False,
  		LOGGER: Log=None
	) -> None:
		self.parameters = parameters
		self.execution_function = execution_function
		self.verbose = verbose
		self.verbose_parameters_sample = verbose_parameters_sample
		self.direct = direct
		self.LOGGER = logging if LOGGER is None else LOGGER
        

def handle_lambda(event: Event) -> LambdaResponse:
	"""Execute a function passing parameters with error handling

	Parameters
	----------
	event : Event
		The Lambda Event

	Returns
	-------
	LambdaResponse
		Lambda Response
	"""
	init = datetime.now()
	parameters = event.parameters
	execution_function = event.execution_function
	logger = event.LOGGER
	parameters_str = str(parameters)
	parameters_sample = f'{parameters_str[:100]} ... {parameters_str[-100:]}'
	if event.verbose:
		if event.verbose_parameters_sample:
			logger.info(f'Start execution with parameters (sample): {parameters_sample}')
		else:			
			logger.info(f'Start execution with parameters: {parameters}')
	function_response = None
	try:
		function_parameters = parameters if event.direct \
      		else Event(parameters=parameters, execution_function=None, LOGGER=logger)
		function_response = execution_function(function_parameters)
		if event.verbose:
			logger.info(f'Succesful execution!')
			logger.info(f'Lambda execution response:')
			response_str = str(function_response)
			log_text = f'{response_str[:100]}'
			if len(response_str) > 100:
				log_text += f' <|:::|> {response_str[-min(len(response_str) - 100, 100):]}'
			logger.info(log_text)
				
	except Exception as exc: 
		if event.verbose_parameters_sample:
			logger.error(f'Execution with parameters (sample): {parameters_sample} generate an exception: {exc}')
		else:
			logger.error(f'Execution with parameters: {parameters} generate an exception: {exc}')
	timeDiff = (datetime.now() - init).total_seconds()
	if event.verbose:
		logger.info(f"Execution ends in {timeDiff}s.")
	return function_response


class ThreadPoolExecutor:
	"""Execute function in parallel threads

	Parameters
	----------
	execution_function : FunctionType
		Function to be executed in concurrents
	worker_data : List[WorkerData]
		The input to be passed for each concurrent
		execution
	max_workers : int, optional
		Max number of parallel threads, by default 5
	worker_id_pattern : FunctionType, optional
		Function to be applied in order to uniquely
		identify a worker, by default lambdax:x
	LOGGER : Log, optional
		The Logger, by default None
	verbose_percentage : float, optional
		The percentage of workers that will
  		be logged, by default 0.1
	stop_count : int, optional
		Stop if number of processed works reach 
		this level, after that, reset counts and start 
		couting again, by default inf.
	stop_computation : FunctionType, optional
		Function to be executed when stop thread pool execution, 
  		by default None.
	del_response_when_stop : bool, optional
		If True, the current response will be deleted after 
  		stop computation executes. If False, the response will 
    	be cumulative till the end, by default False.
	verbose_parameters_sample : bool, optional
		If True, will log only a sample of parameters, 
  		by default True.
	direct : bool, optional
		If True, pass parameters direct to execution function 
  		as arguments, by default False.

	Example
	--------
	Usage example:

	>>> thread_pool_executor = ThreadPoolExecutor(...)
	>>> thread_pool_executor.start()
	>>> thread_pool_executor.get_result()
	<The execution response>
	"""

	def __init__(
		self, 
		execution_function: FunctionType, 
		worker_data: List[WorkerData], 
		max_workers: int=5, 
		worker_id_pattern: FunctionType=lambda worker: worker,
  		LOGGER: Log=None,
		verbose_percentage: float=0.1,
		stop_count: int=inf, 
		stop_computation: FunctionType=None,
  		del_response_when_stop: bool=False,
		verbose_parameters_sample: bool=False,
		direct: bool=False,
	) -> None:
		self.execution_function = execution_function 
		self.worker_data = worker_data 
		self.max_workers = max_workers 
		self.worker_id_pattern = worker_id_pattern 
		if LOGGER is not None:
			self.LOGGER = LOGGER
		else:
			self.LOGGER = Log(log_as_print=True)
		self.verbose_percentage = verbose_percentage
		self.stop_count = stop_count
		self.stop_computation = stop_computation
		self.del_response_when_stop = del_response_when_stop
		self.verbose_parameters_sample = verbose_parameters_sample
		self.direct = direct

	def start(
		self
	) -> None:
		"""Start Thread Pool Execution.
		"""
  
		self.num_of_processed_workers = 0  
		self.started_at = datetime.now()
		self.updated_at = datetime.now()
		self._progress_time = []
  
		if len(self.worker_data) == 0:
			self.LOGGER.info('No data to process...')
			self.result = None
			return None	
  
		self.LOGGER.info('Starting Thread Pool Execution...')  
		thread_executor_response = self._thread_pool_executor(
			self.execution_function, 
			self.worker_data, 
			self.worker_id_pattern,
			self.max_workers,
		)  
		self.LOGGER.info('Thread Pool Execution Finished.')  
		self.result = thread_executor_response

	def _thread_pool_executor(
		self,
		execution_function, 
		worker_data: List[WorkerData], 
		worker_id_pattern: FunctionType,
		max_workers: int=5,
	) -> List[LambdaResponse]:
     
		# define verbose items
		worker_data_len = len(worker_data)
		verbose_items = np.arange(worker_data_len)
		num_elems = math.ceil(len(verbose_items) * self.verbose_percentage)
		verbose_index = np.round(np.linspace(0, worker_data_len - 1, num_elems)).astype(int)
		# verbose_index = verbose_index[1:] if len(verbose_index) > 1 else verbose_index
		verbose_index = verbose_index if verbose_index[-1] == worker_data_len - 1 \
			else verbose_index + [worker_data_len - 1]

		self.TotalWorkerDataLen = worker_data_len
  
		response = []
		worker_data_pack = []
		temp_data = []
		curr_count = 0

		for idx, worker in enumerate(worker_data):
			if idx > 0 and idx % self.stop_count == 0:
				worker_data_pack.append(temp_data)
				temp_data = []
			temp_data.append(worker)
		worker_data_pack.append(temp_data)
		del temp_data

       
		for idx_0, sub_worker_data in enumerate(worker_data_pack):
			curr_response = []
			with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
				futures = {
					executor.submit(
						self._track_progress, 
						{
							'event': Event(
								parameters=row, 
								execution_function=execution_function,
								verbose=(curr_count + idx) in verbose_index,
        						verbose_parameters_sample=self.verbose_parameters_sample,
        						direct=self.direct,
								LOGGER=self.LOGGER,
							),
							'current': curr_count + idx + 1,
						}
					): worker_id_pattern(row) 
					for idx, row in enumerate(sub_worker_data)
				}
				for future in concurrent.futures.as_completed(futures):
					identifier_str = futures[future]
					try:
						data = future.result()
						curr_response.append(data)
					except Exception as exc:
						self.LOGGER.error(f'{identifier_str} generated an exception: {exc}')
      
			curr_count = curr_count + len(sub_worker_data)
   
			if self.stop_computation is not None:
				stop_progress_text = f'({idx_0 + 1}/{len(worker_data_pack)})'
				self.LOGGER.info(f'Stop computation execution {stop_progress_text}...')
				self.stop_computation(curr_response)
				self.LOGGER.info(f'Stop computation execution {stop_progress_text}... Done')
				self.LOGGER.info(f'Complete execution of {curr_count} from {worker_data_len} workers')    
				if self.del_response_when_stop:
					response = []
    
			response = response + curr_response
    
		return response

	def _track_progress(
     	self,	
		data: Dict
	) -> any:

		event = data.get('event')

		lambda_exec_start = datetime.now()
  
  		#########################################
		### Exec Lambda
  		#########################################
		lambda_response = handle_lambda(event)
  		#########################################
    
		lambda_exec_time = (datetime.now() - lambda_exec_start).total_seconds()
  
		total = self.TotalWorkerDataLen

		progress_percentage = progress(
			current=self.num_of_processed_workers, 
			total=total
   		)
  
		if event.verbose:
			self.LOGGER.info(
				f'{progress_percentage}% ({self.num_of_processed_workers:,}/{total:,}) complete.'
			)

			self._progress_time.append(lambda_exec_time)
   
			#########################################
			### Mean of exec time from last 5 workers
			#########################################
			seconds_by_worker = np.array(self._progress_time[-5:]).mean()
			#########################################

			expected_remaining_seconds = math.ceil((total - self.num_of_processed_workers) * seconds_by_worker)
			expected_remaining_seconds = expected_remaining_seconds + 1
   
			self.LOGGER.info(
				f'Expected remaining time: {display_time(expected_remaining_seconds)}'
			)

		return lambda_response

	def get_worker_data(
		self
	) -> WorkerData:
		"""Get worker data

		Returns
		-------
		WorkerData
			The defined worker data
		"""
		return self.worker_data

	def get_result(
		self
	) -> any:
		"""Get result of all concurrent executions

		Returns
		-------
		any
			The result of all concurrent executions.
		"""
		return self.result
