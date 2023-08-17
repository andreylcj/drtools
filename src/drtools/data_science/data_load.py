""" 
This module was created to load and save different types of data. 
For instance: .csv, .txt, .parquet, .gz and so on.

"""


from drtools.file_manager import (
    create_directories_of_path, 
    list_path_of_all_files_inside_directory
)
from typing import Union, List, Callable, Optional
from types import FunctionType
from pandas.core.frame import DataFrame
import pandas as pd
import os
import joblib
import pyarrow.parquet as pq
from enum import Enum
from drtools.logging import FormatterOptions, Logger
from drtools.utils import ExpectedRemainingTimeHandle


class FileType(Enum):
    CSV = "csv", ".csv"
    JSON = "json", ".json"
    
    @property
    def pname(self):
        return self.value[0]
    
    @property
    def extension(self):
        return self.value[1]
    
    
class FileTypeHandler:
    
    def is_csv(self, filepath: str) -> bool:
        return filepath.endswith('.csv') \
            or filepath.endswith('.csv.gz')
    
    def is_parquet(self, filepath: str) -> bool:
        return filepath.endswith('.parquet')
    
    def is_json(self, filepath: str) -> bool:
        return filepath.endswith('.json') \
            or filepath.endswith('.json.gz')
    
    
class BaseDataframeReader(FileTypeHandler):
    def __init__(
        self,
        LOGGER: Logger=None
    ):
        self.LOGGER = LOGGER
        
    def read(self, filepath: str, **kwargs) -> Optional[DataFrame]:
        raise NotImplementedError


class CSVDataframeReader(BaseDataframeReader):
    
    def read(self, filepath: str, **kwargs) -> Optional[DataFrame]:
        return pd.read_csv(filepath, **kwargs)


class ParquetDataframeReader(BaseDataframeReader):
    
    def read(self, filepath: str, **kwargs) -> Optional[DataFrame]:
        chunksize: int = kwargs.get('chunksize', None)
        nrows: int = kwargs.get('nrows', None)
        usecols: List[str] = kwargs.get('usecols', None)
        
        if chunksize is not None:
            return pq.ParquetFile(filepath)
        
        if nrows is not None:
            resp = pq.ParquetFile(filepath)
            df = None
            
            for chunk in resp.iter_batches(batch_size=nrows):
                df = chunk.to_pandas()
                break
            
            if usecols is not None:
                df = df.loc[:, usecols]
                
            return df
        
        else:
            return pd.read_parquet(filepath, columns=usecols)


class JSONDataframeReader(BaseDataframeReader):
    
    def read(self, filepath: str, **kwargs) -> Optional[DataFrame]:
        df = pd.read_json(filepath)        
        usecols = kwargs.get('usecols', None)
        if usecols is not None:
            df = df.loc[:, usecols]
        return df
    
    
class SmartDataframeReader(BaseDataframeReader):
    
    def read(self, filepath: str, **kwargs) -> Optional[DataFrame]:
        
        if self.is_csv(filepath):
            return CSVDataframeReader(
                LOGGER=self.LOGGER
            ).read(
                filepath=filepath,
                **kwargs
            )
        
        elif self.is_parquet(filepath):
            return ParquetDataframeReader(
                LOGGER=self.LOGGER
            ).read(
                filepath=filepath,
                **kwargs
            )
        
        elif self.is_json(filepath):
            return JSONDataframeReader(
                LOGGER=self.LOGGER
            ).read(
                filepath=filepath,
                **kwargs
            )
        
        else:
            return None
    
    
class BaseDirectoryDataframeReader(FileTypeHandler):
    def __init__(
        self,
        LOGGER: Logger=None
    ):
        self.LOGGER = LOGGER
        
    def read(
        self, 
        dirpath: str, 
        process_chunk: Callable=None,
        **kwargs
    ) -> Optional[DataFrame]:
        raise NotImplementedError
        
        
class SmartDirectoryDataframeReader(BaseDirectoryDataframeReader):
        
    def read(
        self, 
        dirpath: str, 
        process_chunk: Callable=None,
        **kwargs
    ) -> Optional[DataFrame]:
        
        filepaths: List[str] = list_path_of_all_files_inside_directory(
            dirpath
        )
        df = None
        
        chunksize: int = kwargs.get('chunksize', None)
        nrows: int = kwargs.get('nrows', None)
        usecols: List[str] = kwargs.get('usecols', None)
        
        for filepath in filepaths:
            curr_df = SmartDataframeReader(
                    LOGGER=self.LOGGER
                ).read(
                    filepath=filepath, 
                    **kwargs
                )
            
            if curr_df is None:
                continue
            
            temp_data = None
            
            if nrows is not None \
            and chunksize is not None:
                raise Exception('Parameter "chunksize" and "nrows" can not be provided at same time.')
            
            if chunksize is not None:
                
                iter_data = None
                if self.is_parquet(filepath):
                    iter_data = curr_df.iter_batches(batch_size=chunksize)
                
                else:
                    iter_data = curr_df
                
                for chunk in iter_data:
                    
                    if temp_data is None:
                        temp_data = process_chunk(chunk)
                        
                    else:
                        temp_data = pd.concat([
                                temp_data,
                                process_chunk(chunk)
                            ], 
                            ignore_index=True
                        )
                
                if usecols is not None:
                    temp_data = temp_data.loc[:, usecols]
                            
            elif nrows is not None:
                df_shape = 0 if df is None else df.shape[0]
                temp_data = curr_df.iloc[:nrows - df_shape]
                
            else:
                temp_data = curr_df                
                    
            if df is None: 
                df = temp_data.copy()
                    
            else: 
                df = pd.concat([
                        df, 
                        temp_data
                    ], 
                    ignore_index=True
                )
                
            if nrows is not None and df.shape[0] >= nrows:
                break
        
        return df


def concat_dir(
    dir: str, 
    outpath: str, 
    verbose: int=100, 
    file_type: FileType=FileType.CSV,
    LOGGER: Logger=Logger(
        formatter_options=FormatterOptions(
            include_thread_name=True,
            include_datetime=True,
            include_level_name=True,
        ),
        default_start=False
    ),
    ignore_error_logs: bool=True,
):
    """Concat all files from directory to single file

    Parameters
    ----------
    dir : str
        The directory path containing files.
    outpath : str
        Path so write output file.
    verbose : int, optional
        Verbose num, by default 100
    file_type : FileType, optional
        Type of files on directory, by default FileType.CSV
    LOGGER : Log, optional
        Logger instance, by default Log( formatter_options=FormatterOptions( IncludeThreadName=True, IncludeDate=True, IncludeLevelName=True, ), default_start=False )
    ignore_error_logs : bool, optional
        If True, all error logs when writting and skipping header 
        from files will be ignored, by default True

    Raises
    ------
    Exception
        If the file_type is not supported.
    """
    all_paths = list_path_of_all_files_inside_directory(dir)
    expected_remaining_time = ExpectedRemainingTimeHandle(total=len(all_paths))
    insert_header = True
    count = 0
    total_paths_len = len(all_paths)
    LOGGER.info('Start concatenating...')
    if file_type is FileType.CSV:
        with open(outpath, 'w') as f:
            for path in all_paths:
                count += 1            
                if count % verbose == 0:
                    LOGGER.debug(f'({(count+1):,}/{total_paths_len:,}) Expected remaining time: {expected_remaining_time.display_time(count-1)}')
                with open(path, 'r') as f1:
                    try:
                        if not insert_header:
                            next(f1)
                        insert_header = False
                        for line in f1:
                            f.write(line)
                    except Exception as exc:
                        if not ignore_error_logs:
                            LOGGER.error(f'Error {exc} on {path}')
    elif file_type is FileType.JSON:
        raise Exception(f"Not implemented.")
    else:
        raise Exception(f"File type {file_type} not supported.")
    LOGGER.info('Start concatenating... Done!')


def save_df(
    dataframe: DataFrame,
    path: str,
    overwrite_if_exists: bool=False,
    **kwargs
) -> None:
    """Smart save of DataFrame.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame to be saved.
    path : str
        Path to save dataframe
    overwrite_if_exists : bool, optional
        If True, overwrite file with same name 
        if exists. If False, return Exception 
        if exist some file on local Database with 
        same filename, by default False
    """
    if not overwrite_if_exists:
        if os.path.exists(path):
            raise Exception(f'Path {path} already exists.')
    
    create_directories_of_path(path)
    
    filename = os.path.basename(path)
    
    if '.csv' in filename:
        dataframe.to_csv(path, **kwargs)
        
    elif '.parquet' in filename:
        dataframe.to_parquet(path, **kwargs)
        
    elif '.json' in filename:
        dataframe.to_json(path, **kwargs)
        
    else:
        raise Exception(f'Extension on {os.path.basename(path)} not allow.')
