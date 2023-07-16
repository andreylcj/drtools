

from typing import Dict
from psycopg2.pool import ThreadedConnectionPool as _ThreadedConnectionPool
from psycopg2.pool import PoolError
        

class Cursor:
    pass


class Connection:
    pass


class ConnectionConfig:
    
    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str=None,
        sslmode: str='require',
        connect_timeout: int=3,
        options: str='-c statement_timeout=3000',
        keepalives: int=1,
        keepalives_idle: int=30,
        keepalives_interval: int=5,
        keepalives_count: int=5
    ) -> None:
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self._password = password
        self.sslmode = sslmode
        self.connect_timeout = connect_timeout
        self.options = options
        self.keepalives = keepalives
        self.keepalives_idle = keepalives_idle
        self.keepalives_interval = keepalives_interval
        self.keepalives_count = keepalives_count
        
    def set_password(self, password):
        self._password = password        
            
    @property
    def info(self) -> Dict:
        return {
            'host': self.host,
            'port': self.port,
            'dbname': self.dbname,
            'user': self.user,
            'sslmode': self.sslmode,
            'connect_timeout': self.connect_timeout,
            'options': self.options,
            'keepalives': self.keepalives,
            'keepalives_idle': self.keepalives_idle,
            'keepalives_interval': self.keepalives_interval,
            'keepalives_count': self.keepalives_count,
        }
    
    def get_password(self) -> str:
        return self._password
    
    @property
    def secret_info(self) -> Dict:
        return {
            **self.info,
            'password': self.get_password(),
        }
    
    @property
    def kwargs(self) -> Dict:
        return {
            **self.info,
            'password': self.get_password(),
        }
    
    @property
    def keepalive_kwargs(self) -> Dict:
        return {
            'keepalives': self.keepalives,
            'keepalives_idle': self.keepalives_idle,
            'keepalives_interval': self.keepalives_interval,
            'keepalives_count': self.keepalives_count,
        }
        
    @property
    def connection_extra_params(self) -> Dict:
        return {
            'sslmode': self.sslmode,
            'connect_timeout': self.connect_timeout,
            'options': self.options,
            'keepalives': self.keepalives,
            'keepalives_idle': self.keepalives_idle,
            'keepalives_interval': self.keepalives_interval,
            'keepalives_count': self.keepalives_count,
        }
        

class ThreadedConnectionPool(_ThreadedConnectionPool):
    
    def _delconn(self, conn, key=None):
        if self.closed:
            raise PoolError("connection pool is closed")
        
        if key is None:
            key = self._rused.get(id(conn))
            if key is None:
                raise PoolError("trying to delete unkeyed connection")
            
        if not self.closed or key in self._used:
            del self._used[key]
            del self._rused[id(conn)]
    
    def delconn(self, conn, key=None):
        """Delete connection from the pool."""
        self._lock.acquire()
        try:
            self._delconn(conn, key)
        finally:
            self._lock.release()
    
    def _addconn(self, replace: bool=False):
        if self.closed:
            raise PoolError("connection pool is closed")
        
        if len(self._used) + len(self._pool) >= self.maxconn and not replace:
            raise PoolError("connection pool is complete")
        
        key = self._getkey()
        return self._connect(key)
    
    def addconn(self, replace: bool=False) -> int:
        """Add new connection to the pool and return connection key."""
        self._lock.acquire()
        try:
            return self._addconn(replace)
        finally:
            self._lock.release()
            
    def _replaceconn(self, conn, key=None):
        self._delconn(conn, key)
        return self._addconn(True)
            
    def replaceconn(self, conn, key=None):
        """Delete connection and add new connection to the pool."""
        self._lock.acquire()
        try:
            return self._replaceconn(conn, key)
        finally:
            self._lock.release()
        

    # def __init__(self, minconn, maxconn, *args, **kwargs):
    #     self._semaphore = Semaphore(maxconn)
    #     super().__init__(minconn, maxconn, *args, **kwargs)

    # def getconn(self, *args, **kwargs):
    #     self._semaphore.acquire()
    #     try:
    #         return super().getconn(*args, **kwargs)
    #     except:
    #         self._semaphore.release()
    #         raise

    # def putconn(self, *args, **kwargs):
    #     try:
    #         super().putconn(*args, **kwargs)
    #     finally:
    #         self._semaphore.release()


class ConnectionPoolConfig:
    def __init__(
        self,
        minconn: int,
        maxconn: int,
    ) -> None:
        self.minconn = minconn
        self.maxconn = maxconn