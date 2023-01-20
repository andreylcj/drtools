""" 
This module was created to handle 
connection in PostgreSQL Databases 
with Python.

"""


from configparser import ConfigParser
from typing import List, Tuple, Union
import psycopg2
import gzip
from drtools.file_manager import path_exist, create_directories_of_path
from drtools.logs import Log
import logging


FetchAll = 'fetch-all'
FetchOne = 'fetch-one'
FetchMany = 'fetch-many'


class Cursor:
    """This class handle psycopg2 connection cursor methods.
    """
    
    def __init__(
        self,
        connection: any,
        fetch_mode: Union[FetchAll, FetchOne, FetchMany]=FetchAll,
        LOGGER: Log=logging
    ) -> None:
        """Handle Psycog2 connection cursor.

        Parameters
        ----------
        connection : _type_
            _description_
        fetch_mode : Union[FetchAll, FetchOne, FetchMany], optional
            _description_, by default FetchAll
        """
        self.LOGGER = LOGGER
        self.LOGGER.info('Initializing cursor...')
        self._cursor = connection.cursor()
        self.fetch_mode = fetch_mode
        self.LOGGER.info('Cursor was successfully initialized!')        
        
    def execute(
        self,
        query: str,
        query_values: Tuple=None,
    ) -> None:
        self.LOGGER.info('Executing query...')
        if query_values is not None:
            self._cursor.execute(query, query_values)
        else:
            self._cursor.execute(query)
        self.LOGGER.info('Query successfully executed...')        
        
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
    
    def description(self) -> any:
        return self._cursor.description    
    
    def copy(
        self, 
        query: str,
        save_path: str
    ) -> None:        
        outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
        if path_exist(save_path):
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


class ConnectionConfig:
    
    def __init__(
        self,
        sslmode='require',
        connect_timeout: int=3,
        options: str='-c statement_timeout=3000',
        keepalives: int=1,
        keepalives_idle: int=15,
        keepalives_interval: int=10,
        keepalives_count: int=3
    ) -> None:
        """Handle connection configuration.
        
        More information about parameters on https://www.postgresql.org/docs/9.3/libpq-connect.html.

        Parameters
        ----------
        sslmode : str, optional
            The SSL mode that will be 
            applied on connection, by default 'require'
        connect_timeout : int, optional
            Maximum wait for connection, in seconds 
            (write as a decimal integer string). 
            Zero or not specified means wait indefinitely. 
            It is not recommended to use a timeout 
            of less than 2 seconds, by default 3
        options : str, optional
            Adds command-line options to send to the 
            server at run-time. For example, 
            setting this to -c geqo=off sets the 
            session's value of the geqo parameter 
            to off, by default '-c statement_timeout=3000'
        keepalives : Union[On, Off], optional
            Controls whether client-side TCP keepalives 
            are used. The default value is 1, meaning 
            on, but you can change this 
            to 0, meaning off, if keepalives are 
            not wanted. This parameter is ignored 
            for connections made via a 
            Unix-domain socket., by default On
        keepalives_idle : int, optional
            Controls the number of seconds of inactivity 
            after which TCP should send a keepalive 
            message to the server. A value of zero uses the 
            system default. This parameter is ignored 
            for connections made via a Unix-domain 
            socket, or if keepalives are 
            disabled. It is only supported on systems where 
            TCP_KEEPIDLE or an equivalent socket option 
            is available, and on Windows; on other systems, it has 
            no effect, by default 15
        keepalives_interval : int, optional
            Controls the number of seconds after which a TCP 
            keepalive message that is not acknowledged by the server 
            should be retransmitted. A value of zero uses the 
            system default. This parameter is ignored for connections 
            made via a Unix-domain socket, or if keepalives are 
            disabled. It is only supported on systems where 
            TCP_KEEPINTVL or an equivalent socket option is 
            available, and on Windows; on other systems, it has no 
            effect, by default 10
        keepalives_count : int, optional
            Controls the number of TCP keepalives that can 
            be lost before the client's connection to the 
            server is considered dead. A value of zero 
            uses the system default. This parameter is 
            ignored for connections made via a Unix-domain 
            socket, or if keepalives are disabled. It is only 
            supported on systems where TCP_KEEPCNT or an 
            equivalent socket option is available; on other 
            systems, it has no effect, by default 3
        """
        args = locals().copy()
        args = {k: v for k, v in args.items() if k != 'self'}
        for k, v in args.items():
            setattr(self, k, v)


class Database:
    
    def __init__(
        self,
        host: str=None,
        dbname: str=None,
        port: int=None,
        user: str=None,
        password: str=None,
        connection_config: ConnectionConfig=ConnectionConfig(),
        LOGGER: Log=logging
    ) -> None:
        self.LOGGER = LOGGER
        self.host = host
        self.dbname = dbname
        self.port = port
        self.user = user
        self.password = password
        self.connection_config = connection_config
        self._connection = None

    def credentials_config(
        self,
        filename: str, 
        section: str,
    ) -> None:
        # create a parser
        parser = ConfigParser()
        # read config file
        parser.read(filename)

        # get section, default to postgresql
        db = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                db[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, filename))
        
        self.host = db.get('host', None)
        self.dbname = db.get('dbname', None)
        self.port = db.get('port', None)
        self.user = db.get('user', None)
        if self.password is None:
            self.password = db.get('password', None)

    def connect(
        self,
    ) -> None:
        """ Connect to the PostgreSQL database server """
        self._connection = None
        try:
            # read connection parameters
            params = {
                'host': self.host,
                'dbname': self.dbname,
                'port': self.port,
                'user': self.user,
                'password': self.password,
            }
            if params['password'] is None:
                raise Exception("You must provice 'password' is required.")            
            
            params = {**params, **self.connection_config.__dict__}
            
            # connect to the PostgreSQL server
            self.LOGGER.info('Connecting to the PostgreSQL database...')
            self._connection = psycopg2.connect(**params)
            self.LOGGER.info('Successful connection!')
            
        except (Exception, psycopg2.DatabaseError) as error:
            self.LOGGER.error(error)                
                
    def execute(
        self, 
        query: str, 
        query_values: Tuple=None,
        fetch_mode: Union[FetchAll, FetchOne, FetchMany]=FetchAll,
        size: int=10,
        save_on: str=None,
    ) -> Tuple[List[str], List[Tuple]]:
        """Execute received query.

        Parameters
        ----------
        query : str
            Text query.
        query_values : Tuple, optional
            Query values, by default None
        fetch_mode : Union[FetchAll, FetchOne, FetchMany], optional
            Fetch mode, by default FetchAll
        size : int, optional
            Fetch size, by default 10
        save_on : str, optional
            Path to save response, by default None

        Returns
        -------
        Tuple[List[str], List[Tuple]]
            Returns the Header and the Body data.
        """
        self._connection = None
        cursor = None
        try:
            self.connect()
            cursor = Cursor(connection=self._connection, fetch_mode=fetch_mode)
            if save_on is None:
                cursor.execute(query, query_values)
                response = cursor.fetch(size)
                colnames = [desc[0] for desc in cursor.description()]
                return colnames, response
            else:
                cursor.copy(query, save_on)
        except (Exception, psycopg2.DatabaseError) as error:
            self.LOGGER.error(error)
        finally:
            if cursor is not None:
                cursor.close()
            self.close()

    def close(self) -> None:
        if self._connection is not None:
                self._connection.close()
                self._connection = None
                self.LOGGER.info('Database connection closed.')