""" 
This module was created to load and save different types of data. 
For instance: .csv, .txt, .parquet, .gz and so on.

"""


from drtools.file_manager import (
    create_directories_of_path, search_by_name_on_directory,
    list_path_of_all_files_inside_directory
)
from typing import Union
from types import FunctionType
from pandas.core.frame import DataFrame
import pandas as pd
import os
import joblib
import pyarrow.parquet as pq


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


def read_as_df(
    file_reference: str, 
    directory: str=None,
    **read_args,
) -> Union[DataFrame, None]:
    """Get DataFrame from name of file inside directory

    Parameters
    ----------
    file_reference : str
        Name of file that will be searched or abs path of file.
    directory : str, optional
        Directory to search for file, by default None.
    read_args : dict, optional
        Arguments that will be passed to read 
        file function, by default {}.

    Returns
    -------
    Union[DataFrame, None]
        If file is found and type is csv or parquet, returns the DataFrame, 
        else return **None**
    """
    file_path = file_reference
    if directory is not None:
        file_names = search_by_name_on_directory(
            file_reference, 
            directory
        )
        if len(file_names) == 0:
            return None
        file_path = os.path.join(directory, file_names[0])
        
    if '.csv' in file_path:
        return pd.read_csv(
            file_path,
            **read_args
        )
        
    elif '.parquet' in file_path:
        chunksize = read_args.get('chunksize', None)
        nrows = read_args.get('nrows', None)
        if chunksize is not None:
            return pq.ParquetFile(file_path)
        elif nrows is not None:
            resp = pq.ParquetFile(file_path)
            df = None
            for chunk in resp.iter_batches(batch_size=nrows):
                df = chunk.to_pandas()
                break
            usecols = read_args.get('usecols', None)
            if usecols is not None:
                df = df.loc[:, usecols]
            return df
        else:
            usecols = read_args.get('usecols', None)
            return pd.read_parquet(
                file_path,
                columns=usecols
            )
    
    elif '.json' in file_path:
        df = pd.read_json(file_path)        
        usecols = read_args.get('usecols', None)
        if usecols is not None:
            df = df.loc[:, usecols]
        return df
    
    
    elif '.joblib' in file_path:
        return joblib.load(file_path)
    
    else:
        return None
    
    
def read_dir_as_df(
    directory_path: str,
    process_chunk: FunctionType=None,
    **read_args,
) -> Union[DataFrame, None]: 
    """Load data of entire directory and load as DataFrame.

    Parameters
    ----------
    directory_path : str
        Path of directory.
    process_chunk : FunctionType
        If chunksize if passed as parameter, process each 
        chunck applying this function. This function must return 
        a DataFrame, by default None.

    Returns
    -------
    DataFrame
        The data loaded as DataFrame.
    """

    items_path = list_path_of_all_files_inside_directory(
        directory_path
    )

    df = None
    
    for item in items_path:
        resp = read_as_df(item, **read_args)
        if resp is None:
            continue
        
        temp_data = None
        chunksize = read_args.get('chunksize', None)
        nrows = read_args.get('nrows', None)
        
        if nrows is not None \
        and chunksize is not None:
            raise Exception('Parameter "chunksize" and "nrows" can not be provided at same time.')
        
        if chunksize is not None:
            if '.parquet' in item:
                for chunk in resp.iter_batches(batch_size=chunksize):
                    chunk = chunk.to_pandas()
                    if temp_data is None:
                        temp_data = process_chunk(chunk)
                    else:
                        temp_data = pd.concat([
                            temp_data,
                            process_chunk(chunk)
                        ], ignore_index=True)
            else:
                for chunk in resp:
                    if temp_data is None:
                        temp_data = process_chunk(chunk)
                    else:
                        temp_data = pd.concat([
                            temp_data,
                            process_chunk(chunk)
                        ], ignore_index=True)
                        
            usecols = read_args.get('usecols', None)
            if usecols is not None:
                temp_data = temp_data.loc[:, usecols]
                        
        elif nrows is not None:
            df_shape = 0 if df is None else df.shape[0]
            temp_data = resp[resp.index < nrows - df_shape]
            
        else:
            temp_data = resp
            
                
        if df is None: 
            if type(temp_data) == dict:
                df = temp_data
            else:
                df = temp_data.copy()
        else: 
            df = pd.concat([df, temp_data], ignore_index=True)
            
        if nrows is not None and df.shape[0] >= nrows:
            break
            
    return df