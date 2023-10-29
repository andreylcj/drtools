

from typing import Any, List
from drtools.logging import Logger, FormatterOptions
from drtools.file_manager import rm_file, create_directories_of_path
from drtools.thread_pool_executor import ThreadPoolExecutor, WorkerData, ThreadConfig
from drtools.retry import RetryConfig, RetryType, RetryHandler
from drtools.database.connection.resources import (
    ConnectionConfig, 
    ThreadedConnectionPool,
    ConnectionPoolConfig
)
from drtools.database.connection.postgresql import PostgreConnection
from psycopg2.pool import AbstractConnectionPool


RawQuery = str
AbsoluteQuery = str


class QueryHandlerResponse:
    def __init__(
        self,
        abs_query: AbsoluteQuery,
        save_path: str,
    ) -> None:
        self.abs_query = abs_query
        self.save_path = save_path

    
class QueryHandler:
    
    def __init__(
        self,
        raw_query: RawQuery,
        LOGGER: Logger=Logger(
            name='QueryHandler',
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_logger_name=True,
                include_level_name=True,
            ),
            default_start=False
        )
    ) -> None:
        self.raw_query = raw_query
        self.LOGGER = LOGGER
        
    def __call__(
        self, 
        raw_query: RawQuery, 
        worker: Any, 
        pool: AbstractConnectionPool
    ) -> QueryHandlerResponse:
        raise Exception("Must be implemented.")


class MassiveExtractionThreadConfig(ThreadConfig):
    def __init__(
        self,
		worker_data: List[WorkerData], 
		max_workers: int=12, 
  		LOGGER: Logger=None,
		verbose: int=1,
		verbose_parameters_sample: bool=False,
		log_traceback: bool=True,
    ):
        super(MassiveExtractionThreadConfig, self).__init__(
            max_workers=max_workers,
            LOGGER=LOGGER,
            verbose=verbose,
            verbose_parameters_sample=verbose_parameters_sample,
            log_traceback=log_traceback,
            direct=True,
        )
        self.worker_data = worker_data


class PostgreMassiveExtraction:
    
    def __init__(
        self,
        query_handler: QueryHandler,
        connection_config: ConnectionConfig,
        thread_config: MassiveExtractionThreadConfig,
        connection_pool_config: ConnectionPoolConfig=None,
        request_retry_config: RetryConfig=RetryConfig(
            name='RequestRetry',
            retry_wait_time=0,
            retry_type=RetryType.STATIC,
        ),
        LOGGER: Logger=Logger(
            name='PostgreMassiveExtraction',
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_logger_name=True,
                include_level_name=True,
            ),
            default_start=False
        )
    ):
        self.query_handler = query_handler
        self.connection_config = connection_config
        self.thread_config = thread_config
        if connection_pool_config is None:
            connection_pool_config = ConnectionPoolConfig(
                minconn=thread_config.max_workers,
                maxconn=thread_config.max_workers+2,
            )
        self.connection_pool_config = connection_pool_config
        self.request_retry_config = request_retry_config
        self.LOGGER = LOGGER
        self._change_thread_exec_function()
    
    def request_data(
        self,
        abs_query: AbsoluteQuery,
        save_path: str,
        pool,
        request_retry_config: RetryConfig,
        worker
    ):
        
        def try_statement(pool, conn, custom_kwargs, **kwargs):
            real_conn = custom_kwargs.get_or('connection', conn)        
            cursor = real_conn.cursor()
            outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(abs_query)            
            create_directories_of_path(save_path)
            with open(save_path,  'w') as f:
                cursor.copy_expert(outputquery, f)
            cursor.close()
            pool.putconn(real_conn)
            
        def fail_finally_statement_before_wait_time(pool, conn, custom_kwargs, exception, **kwargs):
            cursor = custom_kwargs.get_or_none('cursor')
            if cursor is not None:
                cursor.close()
            rm_file(save_path, ignore_if_path_not_exists=True)
            
            # exception_txt = str(exception)
            # if 'SSL SYSCALL error: EOF detected' in exception_txt:
            real_conn = custom_kwargs.get_or('connection', conn)
            self.LOGGER.debug('Replacing connection...')
            connection = pool.replaceconn(real_conn)
            custom_kwargs.add('connection', connection)
            self.LOGGER.debug('Replacing connection... Done!')
        
        request_retry_config.LOGGER.set_level(30)
        
        retry_handler = RetryHandler(
            try_statement=try_statement,
            fail_finally_statement_before_wait_time=fail_finally_statement_before_wait_time,
            log_error_extra_information=str(worker),
            retry_config=request_retry_config
        )
        
        connection = pool.getconn()
        retry_handler.run(pool, connection) 
        self.LOGGER.info(retry_handler.get_statement_info_archive().info)
    
    def _change_thread_exec_function(self):
        
        def _thread_exec_function(worker):
            query_handler_response = self.query_handler(
                raw_query=self.query_handler.raw_query,
                worker=worker, 
                pool=self.get_pool(), 
            )
            self.request_data(
                abs_query=query_handler_response.abs_query,
                save_path=query_handler_response.save_path,
                pool=self.get_pool(),
                request_retry_config=self.request_retry_config,
                worker=worker
            )
        self.thread_config.exec_func = _thread_exec_function    
    
    def get_pool(self):
        return self._pool    
    
    def get_conn(self) -> PostgreConnection:
        postgre_connection = PostgreConnection(
            connection_config=self.connection_config,
            LOGGER=self.LOGGER,
        )
        return postgre_connection
        
    def terminate_connections(
        self,
        usename: str=None
    ) -> Any:
        postgre_connection = self.get_conn()
        return postgre_connection.terminate_connections(usename)
    
    def setup_thread(self):
        self.thread_pool_executor = ThreadPoolExecutor(
                exec_func=self.thread_config.exec_func,
                worker_data=self.thread_config.worker_data,
                thread_config=ThreadConfig(
                    max_workers=self.thread_config.max_workers,
                    LOGGER=self.thread_config.LOGGER,
                    verbose=self.thread_config.verbose,
                    verbose_parameters_sample=self.thread_config.verbose_parameters_sample,
                    direct=self.thread_config.direct,
                    log_traceback=self.thread_config.log_traceback,
                )
        )
    
    def setup_pool(self):
        minconn = self.connection_pool_config.minconn
        maxconn = self.connection_pool_config.maxconn
        self.LOGGER.info(f'Pool connecting...')
        self.LOGGER.info(f'Pool (MinConn | MaxConn) = ({minconn:,} | {maxconn:,}).')
        self._pool = ThreadedConnectionPool(
            minconn=minconn, 
            maxconn=maxconn,
            host=self.connection_config.host,
            port=self.connection_config.port,
            dbname=self.connection_config.dbname,
            user=self.connection_config.user,
            password=self.connection_config.get_password(),
            sslmode=self.connection_config.sslmode,
            connect_timeout=self.connection_config.connect_timeout,
            options=self.connection_config.options,
            keepalives=self.connection_config.keepalives,
            keepalives_idle=self.connection_config.keepalives_idle,
            keepalives_interval=self.connection_config.keepalives_interval,
            keepalives_count=self.connection_config.keepalives_count,
        )
        self.LOGGER.info(f'Pool connecting... Done!')
    
    def start(self):
        self.setup_thread()
        self.setup_pool()
        
        self.thread_pool_executor.start()
        
        self.LOGGER.info('Close all...')
        pool = self.get_pool()
        pool.closeall()
        self.LOGGER.info('Close all... Done!')
        
        self.LOGGER.info('Terminating connections...')
        self.terminate_connections()
        self.LOGGER.info('Terminating connections... Done!')
        