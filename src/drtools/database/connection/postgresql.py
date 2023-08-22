""" 
This module was created to handle 
connection in PostgreSQL Databases 
with Python.

"""


import os
from typing import List, Tuple, Union, Any, Optional
import psycopg2
import gzip
from drtools.file_manager import create_directories_of_path
from drtools.logging import Logger, FormatterOptions
from drtools.database.connection.resources import (
    Cursor, 
    Connection,
    ConnectionConfig
)
from datetime import datetime
from pandas import DataFrame


FetchAll = 'fetch-all'
FetchOne = 'fetch-one'
FetchMany = 'fetch-many'
ColumnNames = List[str]
RequestedData = List[Tuple]


class PostgreCursor(Cursor):
    """This class handle psycopg2 connection cursor methods.
    """
    
    def __init__(
        self,
        connection: any,
        fetch_mode: Union[FetchAll, FetchOne, FetchMany]=FetchAll,
        LOGGER: Logger=Logger(
            name='PostgreCursor',
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_logger_name=True,
                include_level_name=True,
            ),
            default_start=False
        )
    ) -> None:
        """Handle Psycog2 connection cursor."""
        self.LOGGER = LOGGER
        self.LOGGER.info('Initializing cursor...')
        self._cursor = connection.cursor()
        self.fetch_mode = fetch_mode
        self.LOGGER.info('Cursor was successfully initialized!')        
        
    def execute(
        self,
        query: str,
        query_values: Optional[Tuple]=None,
    ) -> None:
        self.LOGGER.info('Executing query...')
        if query_values is not None:
            self._cursor.execute(query, query_values)
        else:
            self._cursor.execute(query)
        self.LOGGER.info('Query successfully executed.')        
        
    def fetch(
        self, 
        size: int=10
    ) -> List[Tuple]:
        if self.fetch_mode == FetchAll:
            response = self._cursor.fetchall()
        elif self.fetch_mode == FetchOne:
            response = self._cursor.fetchone()
        elif self.fetch_mode == FetchMany:
            assert size is not None, 'If fetch == FetchMany, you need provide "size" value.'
            response = self._cursor.fetchmany(size)
        else:
            raise Exception('Invalid "fetch" mode.')
        return response
    
    @property
    def description(self) -> any:
        return self._cursor.description    
    
    def copy(
        self, 
        query: str,
        save_path: str
    ) -> None:        
        outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
        if os.path.exists(save_path):
            raise Exception(f'Path {save_path} already exists.')
        create_directories_of_path(save_path)
        if save_path.endswith('.gz'):
            with gzip.open(save_path,  'w') as f:
                self._cursor.copy_expert(outputquery, f)
        else:
            with open(save_path, 'w') as f:
                self._cursor.copy_expert(outputquery, f)    
    
    def close(self) -> None:
        self._cursor.close()
        self._cursor = None


class PostgreConnection(Connection):
    def __init__(
        self,
        connection_config: ConnectionConfig,
        LOGGER: Logger=Logger(
            name='PostgreConnection',
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_logger_name=True,
                include_level_name=True,
            ),
            default_start=False
        )
    ) -> None:
        if not connection_config.get_password():
            raise Exception('Password must be provided')
        self.connection_config = connection_config
        self.LOGGER = LOGGER
        self._connection = None
        self.executing = False
        self.exec_details = []
        self.started_at = datetime.now()
        self.last_connection_at = None
        
        
    def connect(self) -> None:
        """ Connect to the PostgreSQL database server """
        self._connection = None
        try:
            # read connection parameters
            params = {
                'host': self.connection_config.host,
                'dbname': self.connection_config.dbname,
                'port': self.connection_config.port,
                'user': self.connection_config.user,
                'password': self.connection_config.get_password(),
            }
            if params['password'] is None:
                raise Exception("You must provice 'password' is required.")            
            
            params = {**params, **self.connection_config.connection_extra_params}
            
            # connect to the PostgreSQL server
            self.LOGGER.info('Connecting to the PostgreSQL database...')
            self._connection = psycopg2.connect(**params)
            self.last_connection_at = datetime.now()
            self.LOGGER.info('Successful connection!')
            
        except (Exception, psycopg2.DatabaseError) as error:
            self.LOGGER.error(error) 
    
    
    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self.LOGGER.info('Database connection closed.')
    
    
    def get_conn(self):
        return self._connection
    
    
    def refresh(self):
        self.close()
        self.connect()
    
    
    def execute(
        self, 
        query: str, 
        query_values: Tuple=None,
        fetch_mode: Union[FetchAll, FetchOne, FetchMany]=FetchAll,
        size: int=10,
        save_on: str=None,
        return_as_dataframe: bool=False,
    ) -> Union[
        Tuple[ColumnNames, RequestedData],
        DataFrame
    ]:
        self.executing = True
        self.exec_started_at = datetime.now()
        cursor = None
        try:
            # self.connect()
            cursor = PostgreCursor(
                connection=self._connection, 
                fetch_mode=fetch_mode,
                LOGGER=self.LOGGER
            )
            if save_on is None:
                cursor.execute(query, query_values)
                response = cursor.fetch(size)
                colnames = [desc[0] for desc in cursor.description]
                if return_as_dataframe:
                    return DataFrame(response, columns=colnames)
                else:
                    return colnames, response
            else:
                cursor.copy(query, save_on)
        except (Exception, psycopg2.DatabaseError) as error:
            self.LOGGER.error(error)
        finally:
            if cursor is not None:
                cursor.close()
        self.executing = False
        self.exec_details.append({
            'started_at': self.exec_started_at,
            'finished_at': datetime.now(),
        })
    
    
    def terminate_connections(
        self,
        usename: str=None
    ) -> Any:
        if usename is None:
            usename = self.connection_config.user
        query = f"""SELECT pg_terminate_backend(pid) FROM pg_stat_activity
        WHERE pid <> pg_backend_pid()
        and usename = '{usename}'
        """
        self.connect()
        terminate_response = self.execute(query)
        self.LOGGER.debug(f'Terminate Connections Response: {terminate_response}')
        self.close()
        return terminate_response
    
    
    @property
    def execution_count(self) -> int:
        return len(self.exec_details)