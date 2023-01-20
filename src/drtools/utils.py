""" 
This module was created to define utils functions that can be used
in many situations.

"""

import os
from datetime import datetime
import re
import platform
from types import FunctionType, LambdaType
from typing import Dict, List, Tuple, Union, Optional, TypedDict
import json
import numpy as np
import pandas as pd
from pandas import DataFrame


def progress(
    current: int, 
    total: int, 
) -> int:
    """Get percentage of progress of some work

    Parameters
    ----------
    current : int
        Current number of points that was 
        be done until this moment.
    total : int
        Toal number of progress points

    Returns
    -------
    int
        The percentage between 0 and 100
    """
    progress = int(round(current * 100 / total, 0))
    if progress >= 100 \
        or (progress != 100 and current == total): 
        progress = 100
    return progress


def flatten(
    dict_: Dict, 
    separator: str='.'
) -> Dict:
    """Flatten dict.

    Parameters
    ----------
    dict_ : Dict
        Ordinary Dict to be flattening
    separator : str, optional
        Separator of dict depth keys when flattening, by default '.'

    Returns
    -------
    Dict
        The flatten dict
    """
    def _flatten(_dict_: Dict, _separator: str, parent_key: str) -> Dict:
        items = []
        for k, v in _dict_.items():
            new_key = parent_key + _separator + k if parent_key else k
            if type(v) == dict:
                items.extend(
                    _flatten(v, _separator=_separator, parent_key=new_key).items()
                )
            else:
                new_v = None
                if type(v) == list:
                    new_v = []
                    for item in v:
                        if type(item) == dict:
                            val = _flatten(item, _separator, parent_key='')
                        else:
                            val = item
                        new_v.append(val)
                items.append((new_key, new_v or v))
        return dict(items)
    result = _flatten(dict_, separator, parent_key='')
    return result
  
  
def re_flatten(
    dict_: Dict, 
    separator: str = '.'
) -> Dict:
    """Re flatten Dict after flatten operation has been applied to dict

    Parameters
    ----------
    dict_ : Dict
        Flatten dict
    separator : str, optional
        Key separator when flatten operation was applied, by default '.'

    Returns
    -------
    Dict
        The re-flatten dict
    """
    def _re_flatten(_dict_: Dict, _separator: str, depth: int=0) -> Dict:
        res_obj = {}
        again = False
        for key in _dict_:
            key_splited = key.split(_separator)
            if len(key_splited) > 1:
                l1 = _separator.join(key_splited[:-1])
                l2 = key_splited[-1]
                res_obj[l1] = res_obj.get(l1, {}) if depth == 0 else _dict_.get(l1, {})
                res_obj[l1][l2] = _dict_[key]
                again = True
            else:
                val = _dict_[key]
                """ if type(val) == list:
                    val = []
                    for item in _dict_[key]:
                        if type(item) == dict:
                            val.append(_re_flatten(item, _separator, depth=0))
                        else:
                            val.append(item) """
                res_obj[key] = val
        if again:
            res_obj = _re_flatten(res_obj, _separator, depth=depth + 1)
        return res_obj
    result = _re_flatten(dict_, separator, depth=0)
    return result


def is_float(
    my_str: str
) -> bool:
    """Verify if string is float format.

    Parameters
    ----------
    my_str : str
        Input string

    Returns
    -------
    bool
        True if string is float, else, False
    """
    
    resp = False
    try:
        float(my_str)
        resp = True
    except Exception:
        resp = False
    return resp


def is_int(
    my_str: str
) -> bool:
    """Verify if string is int format.

    Parameters
    ----------
    my_str : str
        Input string

    Returns
    -------
    bool
        True if string is int, else, False
    """
    
    resp = False
    try:
        int(my_str)
        resp = True
    except Exception:
        resp = False
    return resp


def list_ops(
    list1: List,
    list2: List,
    ops: str='difference'
) -> List:
    """Realize operation between two lists.
    
    Difference:
    - Get element which exists in 'list1' but not exist in 'list2'.

    Parameters
    ----------
    list1 : List
        List one.
    list2 : List
        List two.
    ops : str
        The desired operation to be performed 
        on list, by default 'difference'.

    Returns
    -------
    List
        Returns the result of the selected operation.
    """
    
    if ops == 'difference':
        s = set(list2);
        return [x for x in list1 if x not in s]
    elif ops == 'intersection':
        s = set(list2);
        return [x for x in list1 if x in s]
    elif ops == 'union':
        return list_ops(list1, list2, ops='difference') + list2
    else:
        raise Exception('Invalid "ops" option.')

# *****************************
# [DEPRECATED]
# *****************************
List1 = int
List2 = int
def list_difference(
    list1: List,
    list2: List,
    axis: Union[List1, List2]=1
) -> List:
    """Get element which exists in 'list1' but not exist in 'list2' and
    get elements which exists in 'list1' but not exist in 'list2'.

    Parameters
    ----------
    list1 : List
        List one.
    list2 : List
        List two. 
    axis : Union[List1, List2], optional
        Can be only 1 or 2. 
        If 1, return values that is only in list 1,
        if 2, return values that is only in list 2,
        by default 1

    Returns
    -------
    ListDifference
        Returns the elements which exists in 'list1' and not exists 
        in 'list2' and elements which exists in 'list1' and not exists 
        in 'list2'
    """
    
    if axis == 1:
        return [item for item in list1 if item not in list2]
    elif axis == 2:
        return [item for item in list2 if item not in list1]
    else:
        raise Exception('Invalid "axis" option.')


def camel_to_snake(
    name: str
) -> str:
    """Transform camel case to snake case

    Parameters
    ----------
    name : str
        Camel case name

    Returns
    -------
    str
        Snake case name
    """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def to_title(
    text: str,
    upper_initials: bool=False
) -> str:
    """Transforms text to title
    
    Title remains first letter in capslock and lower for the remaining. 
    Replaces '_' by ' '. If in camel case, add space.

    Parameters
    ----------
    text : str
        Text to be transformed in title
    upper_initials : bool, optional
        If True, initials of all words will be in Caps Lock, 
        If False, only the first letter of the received string will be in Caps Lock, 
        by default False.

    Returns
    -------
    str
        Title text
    """
    text = camel_to_snake(text)
    text = text.replace('_', ' ')
    if not upper_initials:
        return text[0].upper() + text[1:]
    else:
        text = text.split(' ')
        text = [x[0].upper() + x[1:] for x in text]
        text = ' '.join(text)
        return text
    

def hightlight_header(
    title: str,
    break_line_after: int=1
) -> str:
    """Generate hightlight title to print in console

    Parameters
    ----------
    title : str
        Text on header
    break_line_after : int, optional
        Num of break lines to separe title from text, by default 1

    Returns
    -------
    str
        Hightlighted header
    """
    real_title = f'!*** {title} ***!'
    hightlight = f'!{"*" * (len(real_title) - 2)}!'
    return f'{hightlight}\n{real_title}\n{hightlight}' + ("\n" * break_line_after)


# *******************************************
# deprecated ********************************
# *******************************************
def json_pp_message(
    dict_data: dict, 
    sort_keys: bool=True, 
    indent: int=4
) -> str:
    """Get json in a pretty format

    Parameters
    ----------
    dict_data : dict
        Raw data
    sort_keys : bool, optional
        Sort keys of dict, by default True
    indent : int, optional
        Indent desired in message, by default 4

    Returns
    -------
    str
        Json message with pretty format
    """
    return json.dumps(dict_data, sort_keys = sort_keys, indent = indent)


# *******************************************
# deprecated ********************************
# *******************************************
def json_pp(
    dict_data: dict,
    sort_keys: bool=False,
    indent: int=4
) -> None:
    """Pretty print of json data

    Parameters
    ----------
    dict_data : dict
        Raw json data
    sort_keys : bool, optional
        Sort keys of json, by default True
    indent : int, optional
        Indentation of json message, by default 4
    """
    print(
        json_pp_message(
            dict_data=dict_data,
            sort_keys=sort_keys,
            indent=indent,
        )
    )
    
    
def get_os_name() -> str:
    """Get operational system

    Returns
    -------
    str
        Name of operational system
    """
    return platform.system()
    
    
def join_path(
    *args: Tuple[str],
) -> str:
    """Join multiple paths.
    
    Consider special cases, like when one of the paths
    starts with '/'.

    Returns
    -------
    str
        Joined path.
    """
    path = None
    for index, arg in enumerate(args):
        if index != 0 and arg and arg[0] == "/":
            arg = arg[1:]
        if index == 0: 
            path = os.path.abspath(os.path.join(arg))
        else: 
            path = os.path.abspath(os.path.join(path, arg))
    return path


def display_time(
    seconds: int, 
    granularity: int=2
) -> str:
    """Display time based on granularity by converting seconds.
    
    Convert seconds to weeks, days, hours, minutes and seconds.
    
    Parameters
    ----------
    seconds : int
        Number of seconds.
    granularity : int, optional
        Granularity of response, 
        by default 2.

    Returns
    -------
    str
        The corresponding time based on granularity.
    """
    
    intervals = (
        ('months', 604800),  # 60 * 60 * 24 * 30
        ('weeks', 604800),  # 60 * 60 * 24 * 7
        ('days', 86400),    # 60 * 60 * 24
        ('hours', 3600),    # 60 * 60
        ('minutes', 60),
        ('seconds', 1),
    )
    
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])


def isnan(
    value: any
) -> bool:
    """Check if inputed value is nan.

    Parameters
    ----------
    value : any
        Value to check if is nan.

    Returns
    -------
    bool
        If is nan, returns True, 
        else, returns False
    """
    resp = False
    has_except = False
    
    try:
        resp = pd.isna(value)
    except:
        resp = False
        has_except = True
        
    if has_except:
        has_except = False
        try:
            resp = np.isnan(value)
        except:
            resp = False
            has_except = True
            
    if has_except:
        has_except = False
        resp = False
        
    return resp


def iso_str(time_sep='-') -> str:
    """Get time as string. Useful to generate unique file names

    Parameters
    ----------
    time_sep : str, optional
        Separtor of hours, minutes, seconds and 
        microseconds, by default '-'

    Returns
    -------
    str
        String representing now date.
    """
    return datetime.now().strftime(f"%Y-%m-%dT%H:%M:%S.%fZ")


class ValueRestrictionsAsJson(TypedDict):
    bigger_than: Optional[float]
    bigger_or_equal_than: Optional[float]
    smaller_than: Optional[float]
    smaller_or_equal_than: Optional[float]
    equals_to: Optional[float]
    not_equals_to: Optional[float]


class ValueRestrictions:
    """Handle restrictions in values, especially useful when restrict 
    DataFrame numerical values.
    """

    @staticmethod
    def BIGGER_THAN(desired_compare_value, reference_value): 
        return desired_compare_value > reference_value

    @staticmethod
    def BIGGER_OR_EQUAL_THAN(desired_compare_value, reference_value): 
        return desired_compare_value >= reference_value

    @staticmethod
    def SMALLER_THAN(desired_compare_value, reference_value): 
        return desired_compare_value < reference_value

    @staticmethod
    def SMALLER_OR_EQUAL_THAN(desired_compare_value, reference_value): 
        return desired_compare_value <= reference_value

    @staticmethod
    def EQUALS_TO(desired_compare_value, reference_value): 
        return desired_compare_value == reference_value

    @staticmethod
    def NOT_EQUALS_TO(desired_compare_value, reference_value): 
        return desired_compare_value != reference_value
    
    def __init__(
        self,
        bigger_than: float=None,
        bigger_or_equal_than: float=None,
        smaller_than: float=None,
        smaller_or_equal_than: float=None,
        equals_to: float=None,
        not_equals_to: float=None,
        extra_restrictions: List[Union[FunctionType, LambdaType]]=None
    ) -> None:
        self.bigger_than = bigger_than
        self.bigger_or_equal_than = bigger_or_equal_than
        self.smaller_than = smaller_than
        self.smaller_or_equal_than = smaller_or_equal_than
        self.equals_to = equals_to
        self.not_equals_to = not_equals_to
        self.extra_restrictions = extra_restrictions

    def as_json(self) -> dict:
        resp = {}
        if self.bigger_than != None: 
            resp["bigger_than"] = self.bigger_than
        if self.bigger_or_equal_than != None: 
            resp["bigger_or_equal_than"] = self.bigger_or_equal_than
        if self.smaller_than != None: 
            resp["smaller_than"] = self.smaller_than
        if self.smaller_or_equal_than != None: 
            resp["smaller_or_equal_than"] = self.smaller_or_equal_than
        if self.equals_to != None: 
            resp["equals_to"] = self.equals_to
        if self.not_equals_to != None: 
            resp["not_equals_to"] = self.not_equals_to
        return resp

    def verify(
        self,
        value
    ) -> bool:
        resp = []
        if self.bigger_than != None: 
            resp.append(self.BIGGER_THAN(value, self.bigger_than))
        if self.bigger_or_equal_than != None: 
            resp.append(self.BIGGER_OR_EQUAL_THAN(value, self.bigger_or_equal_than))
        if self.smaller_than != None: 
            resp.append(self.SMALLER_THAN(value, self.smaller_than))
        if self.smaller_or_equal_than != None: 
            resp.append(self.SMALLER_OR_EQUAL_THAN(value, self.smaller_or_equal_than))
        if self.equals_to != None: 
            resp.append(self.EQUALS_TO(value, self.equals_to))
        if self.not_equals_to != None: 
            resp.append(self.NOT_EQUALS_TO(value, self.not_equals_to))
        resp = [x for x in resp if not x]
        if len(resp) > 0: 
            resp = False
        else: 
            resp = True

        if self.extra_restrictions != None:
            extra_resp = []
            for extra_restriction in self.extra_restrictions:
                extra_resp.append(extra_restriction(value)) 
            extra_resp = [x for x in extra_resp if not x]
            if len(extra_resp) > 0: extra_resp = False
            else: extra_resp = True
            resp = resp and extra_resp

        return resp
    
    def restrict_df(
        self,
        df: DataFrame,
        cols: Union[str, List[str]],
    ) -> DataFrame:
        
        real_cols = cols
        if type(cols) == str:
            real_cols = [cols]
        
        if self.bigger_than is not None: 
            df = df[np.all(self.BIGGER_THAN(df[real_cols], self.bigger_than), axis=1)]
        if self.bigger_or_equal_than is not None: 
            df = df[np.all(self.BIGGER_OR_EQUAL_THAN(df[real_cols], self.bigger_or_equal_than), axis=1)]
        if self.smaller_than is not None: 
            df = df[np.all(self.SMALLER_THAN(df[real_cols], self.smaller_than), axis=1)]
        if self.smaller_or_equal_than is not None: 
            df = df[np.all(self.SMALLER_OR_EQUAL_THAN(df[real_cols], self.smaller_or_equal_than), axis=1)]
        if self.equals_to is not None: 
            df = df[np.all(self.EQUALS_TO(df[real_cols], self.equals_to), axis=1)]
        if self.not_equals_to is not None: 
            df = df[np.all(self.NOT_EQUALS_TO(df[real_cols], self.not_equals_to), axis=1)]
        return df
    
    def restrict_numpy(
        self,
        matrix: np.array,
        cols_indice: Union[int, List[int]],
    ) -> DataFrame:
        
        real_cols_indice = cols_indice
        if type(cols_indice) == int:
            cols_indice = [cols_indice]
        
        if self.bigger_than is not None: 
            # if is_int:
            #     matrix = matrix[
            #         np.nonzero(self.BIGGER_THAN(matrix[:, col_indice], self.bigger_than))
            #     ]
            # else:
            matrix = matrix[
                np.nonzero(np.all(
                    self.BIGGER_THAN(matrix[:, real_cols_indice], self.bigger_than),
                    axis=1
                ))
            ]                
        if self.bigger_or_equal_than is not None: 
            # if is_int:
            #     matrix = matrix[
            #         np.nonzero(self.BIGGER_OR_EQUAL_THAN(matrix[:, col_indice], self.bigger_or_equal_than))
            #     ]
            # else:
            matrix = matrix[
                np.nonzero(np.all(
                    self.BIGGER_OR_EQUAL_THAN(matrix[:, real_cols_indice], self.bigger_or_equal_than),
                    axis=1
                ))
            ]
        if self.smaller_than is not None: 
            # if is_int:
            #     matrix = matrix[
            #         np.nonzero(self.SMALLER_THAN(matrix[:, col_indice], self.smaller_than))
            #     ]
            # else:
            matrix = matrix[
                np.nonzero(np.all(
                    self.SMALLER_THAN(matrix[:, real_cols_indice], self.smaller_than),
                    axis=1
                ))
            ]
        if self.smaller_or_equal_than is not None: 
            # if is_int:
            #     matrix = matrix[
            #         np.nonzero(self.SMALLER_OR_EQUAL_THAN(matrix[:, col_indice], self.smaller_or_equal_than))
            #     ]
            # else:
            matrix = matrix[
                np.nonzero(np.all(
                    self.SMALLER_OR_EQUAL_THAN(matrix[:, real_cols_indice], self.smaller_or_equal_than),
                    axis=1
                ))
            ]
        if self.equals_to is not None: 
            # if is_int:
            #     matrix = matrix[
            #         np.nonzero(self.EQUALS_TO(matrix[:, col_indice], self.equals_to))
            #     ]
            # else:
            matrix = matrix[
                np.nonzero(np.all(
                    self.EQUALS_TO(matrix[:, real_cols_indice], self.equals_to),
                    axis=1
                ))
            ]
        if self.not_equals_to is not None: 
            # if is_int:
            #     matrix = matrix[
            #         np.nonzero(self.NOT_EQUALS_TO(matrix[:, col_indice], self.not_equals_to))
            #     ]
            # else:
            matrix = matrix[
                np.nonzero(np.all(
                    self.NOT_EQUALS_TO(matrix[:, real_cols_indice], self.not_equals_to),
                    axis=1
                ))
            ]
        return matrix

    def initialize_from_dict(
        self,
        value_restrictions_dict: ValueRestrictionsAsJson    
    ) -> None:
        self.bigger_than = value_restrictions_dict.get('bigger_than', None)
        self.bigger_or_equal_than = value_restrictions_dict.get('bigger_or_equal_than', None)
        self.smaller_than = value_restrictions_dict.get('smaller_than', None)
        self.smaller_or_equal_than = value_restrictions_dict.get('smaller_or_equal_than', None)
        self.equals_to = value_restrictions_dict.get('equals_to', None)
        self.not_equals_to = value_restrictions_dict.get('not_equals_to', None)
        self.extra_restrictions = value_restrictions_dict.get('extra_restrictions', None)