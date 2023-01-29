""" 
This module was created to manage files and directories
of system.

"""

import os
from typing import List
import drtools.utils as Utils
import json
import gzip


def path_exist(
    path: str
) -> bool: 
    """Verify if path exists.

    Parameters
    ----------
    path : str
        Path to be verified

    Returns
    -------
    bool
        If True, path exists, if False, path does
        not exists.
    """
    return os.path.exists(path)


def split_path(
    path: str
) -> List[str]:
    """Handle path split by of os.

    Parameters
    ----------
    path : str
        Path to be splited

    Returns
    -------
    List[str]
        List with pieces of path.
    """
    if Utils.get_os_name() == "Windows": return path.split("\\")
    elif Utils.get_os_name() == "Linux": return path.split("/")
    elif Utils.get_os_name() == "Darwin": return path.split("\\")


def move_file(
    current_path: str,
    destination_path: str,
) -> None:
    """Move file handling when file already exists.

    Parameters
    ----------
    current_path : str
        Current path location
    destination_path : str
        Destination path

    Raises
    ------
    Exception
        If current path does not exists
    Exception
        If destination path already exists
    """
    
    if not path_exist(current_path):
        raise Exception('Current path does not exists.')
    if path_exist(destination_path):
        raise Exception('Destination path already exists.')   
    
    create_directories_of_path(destination_path)
    os.rename(current_path, destination_path)


def get_files_path(
    parent_path: str, 
    content_arr: List[str]=[],
    ignore_files: List[str]=[],
    ignore_folders: List[str]=[],
    ignore_if_contain_some_of_the_words: List[str]=[]
) -> List:
    """Return array containing all abs path of files inside directory.

    Parameters
    ----------
    parent_path : str
        Directory to search files.
    content_arr : List[str], optional
        Array of content of parent path directory, by default []
    ignore_files : List[str], optional
        Ignore files if name equal some of this list, by default [] 
    ignore_folders : List[str], optional
        Ignore folders if name equal some of this list, by default []
    ignore_if_contain_some_of_the_words : List[str], optional
        Ignore folders and files if name cotain some of 
        the names presented on this list, by default []

    Returns
    -------
    List
        List of files path inside parent path directory.
    """
    filesPathArr = []
    ignore_cuz_content_name_have_some_of_the_words = False
    for contentName in content_arr:
        ignore_words_len = len(ignore_if_contain_some_of_the_words)
        for index1, word in enumerate(ignore_if_contain_some_of_the_words):
            if word in contentName:
                ignore_cuz_content_name_have_some_of_the_words = True
                break
            elif index1 == ignore_words_len - 1:
                ignore_cuz_content_name_have_some_of_the_words = False
        if len(contentName.split(".")) > 1:
            if contentName not in ignore_files \
            and not ignore_cuz_content_name_have_some_of_the_words:
                filesPathArr.append(
                    Utils.join_path(parent_path, contentName)
                )
        elif contentName not in ignore_folders \
        and not ignore_cuz_content_name_have_some_of_the_words:
            nextParentPath = Utils.join_path(parent_path, contentName)
            nextContentArr = os.listdir(nextParentPath)
            filesPathArr = filesPathArr \
                + get_files_path(
                    nextParentPath, 
                    nextContentArr, 
                    ignore_files, 
                    ignore_folders, 
                    ignore_if_contain_some_of_the_words
                )
    return filesPathArr


def list_path_of_all_files_inside_directory(
    root_directory_path: str,
    ignore_files: List[str]=[],
    ignore_folders: List[str]=[],
    ignore_if_contain_some_of_the_words: List[str]=[]
) -> List:
    """Return array containing all abs path inside some directory.

    Parameters
    ----------
    root_directory_path : str
        Directory abs path. Search for files inside this directory
    ignore_files : List[str], optional
        Ignore files if name equal some of this list, by default []
    ignore_folders : List[str], optional
        Ignore folders if name equal some of this list, by default []
    ignore_if_contain_some_of_the_words : List[str], optional
        Ignore folders and files if name cotain some of 
        the names presented on this list, by default []

    Returns
    -------
    List
        List of files path inside parent path directory.
    """
    try:
      dataDirectoryContent = os.listdir(root_directory_path)
    except:
      dataDirectoryContent = []
    itemsPath = []
    if len(dataDirectoryContent) > 0:
      itemsPath = get_files_path(
            root_directory_path, 
            dataDirectoryContent, 
            ignore_files, 
            ignore_folders, 
            ignore_if_contain_some_of_the_words
        )
    return itemsPath


def search_by_name_on_directory(
    name: str, 
    directory: str,
) -> List[str]:
    """Search files that contain substring on name inside directory

    Parameters
    ----------
    name : str
        Name of file that will be searched
    dir : str
        Directory to search for file

    Returns
    -------
    List[str]
        All file names that was found
    """
    content = os.listdir(directory)
    result = [filename for filename in content if name in filename]
    return result
    
    
def create_directories_of_path(
    path: str
) -> None:
    """Create all directories of path.

    Parameters
    ----------
    path : str
        Desired path.
        
    Returns
    -----
    None
        None is returned
    """
    
    split_path = None
    
    if Utils.get_os_name() == "Windows": 
        split_path = path.split("\\")
    elif Utils.get_os_name() == "Linux": 
        split_path = path.split("/")
    elif Utils.get_os_name() == "Darwin": 
        split_path = path.split("\\")
        
    temp_path = None
    
    for index, env_path in enumerate(split_path):
        if not env_path: 
            continue
        if len(env_path.split(".")) > 1 and index == len(split_path) - 1: 
            break
        if temp_path == None: 
            if Utils.get_os_name() == "Windows": 
                temp_path = env_path + '\\'
                continue
            elif Utils.get_os_name() == "Linux": 
                temp_path = "/"
        temp_path = Utils.join_path(temp_path, env_path)
        if not os.path.exists(temp_path): 
            try:
                os.makedirs(temp_path, exist_ok=True)
            except Exception as e:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(e).__name__, e.args)
                print("Error: " + str(e))
                print(message)


def save_json(
    data: any, 
    path: str,
    indent: int=4,
    sort_keys: bool=False
) -> None:
    """Save JSON file.

    Parameters
    ----------
    data : any
        Json data
    path : str
        Destination path of file that will be saved
    indent : int, optional
        Indentation of JSON, by default 4
    sort_keys : bool, optional
        If keys of JSON will be sorted, by default False
    
    Returns
    ------
    None
        None is returned
        
    Raises
    ------
    Exception
        If path already exists.
    """
    
    if os.path.exists(path):
        raise Exception('Path already exists.')
    
    create_directories_of_path(path)
    
    if path.endswith('.gz'):
        with gzip.open(path, 'wt', encoding="ascii") as zipfile:
            json.dump(data, zipfile)
    else:
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent, sort_keys=sort_keys)
                
        
def load_json(
    path: str,
) -> any:
    """Load JSON file.

    Parameters
    ----------
    path : str
        Path of file that will be loaded.
    
    Returns
    ------
    None
        None is returned
        
    Raises
    ------
    Exception
        If path does not exists.
    """
    if not os.path.exists(path):
        raise Exception('Path does not exists.')
    methods_by_type_of_file = {
        '.gz': {'open': gzip.open,'apply_on_data': lambda data: data.decode('utf-8')},
        'others': {'open': open,'apply_on_data': lambda data: data},
    }
    extension_path = '.' + path.split('.')[-1]
    methods = methods_by_type_of_file.get(
        extension_path, methods_by_type_of_file['others']
    )
    open_method = methods['open']
    apply_on_data_method = methods['apply_on_data']
    try:
        with open_method(path, "r") as f:
            data = f.read()
            data = json.loads(
                apply_on_data_method(data)
            )
    except json.JSONDecodeError as exc:
        data = []
        for line in open_method(path, 'r'):
            try:
                row = json.loads(line)
                data.append(row)
            except Exception as exc:
                pass
    return data
    
    
def rm_file(
    path: str, 
    ignore_if_path_not_exists: bool=False
) -> None: 
    """Delete file

    Parameters
    ----------
    path : str
        Abs path of file
    ignore_if_path_not_exists : bool, optional
        If True, if path not exists, do not raise Exception, 
        if False, if path not exists, raise Exception, 
        by default False
    """
    if ignore_if_path_not_exists:
      if os.path.exists(path):
        os.remove(path)
    else:
      os.remove(path)