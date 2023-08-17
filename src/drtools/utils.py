""" 
This module was created to define utils functions that can be used
in many situations.

"""

import os
from datetime import datetime
from dateutil import parser, tz
import re
import platform
import json
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import (
    Any, Type, TypedDict, Callable,
    Tuple, Dict, List, Union, Optional,
    get_origin, get_args
)
import math
from enum import Enum
from copy import deepcopy


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
    separator: str='.',
    flatten_list: bool=False
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
            if isinstance(v, dict):
                items.extend(
                    _flatten(v, _separator=_separator, parent_key=new_key).items()
                )
            else:
                if isinstance(v, list):
                    new_v = []
                    for idx, item in enumerate(v):
                        if isinstance(item, dict):
                            val = _flatten(item, _separator, parent_key='')
                        else:
                            val = item
                            
                        if flatten_list:
                            items.append((f'{new_key}{_separator}$idx:{idx}', val))
                        else:
                            new_v.append(val)
                    if not flatten_list:
                        items.append((new_key, new_v))
                else:
                    items.append((new_key, v))
        return dict(items)
    result = _flatten(dict_, separator, parent_key='')
    if flatten_list:
        result = _flatten(result, separator, parent_key='')
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
        return list(list1) + list_ops(list2, list1, ops='difference')
    else:
        raise Exception('Invalid "ops" option.')


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


BASE_TYPES = [str, float, int, bool]


class CustomTreatment:
    def __init__(
        self,
        fullname: str,
        func: Callable
    ) -> None:
        self.fullname = fullname
        self.func = func
        
    def apply(self, class_):
        return self.func(class_)


def convert_keys_to_strings(obj, depth: int=0):
    
    response = None
    iter_items = []
    
    is_dict = lambda x: isinstance(x, dict)
    has_iter = lambda x: hasattr(x, "__iter__") and not isinstance(x, str)
    
    obj_is_dict = is_dict(obj)
    obj_has_iter = has_iter(obj)
    
    if depth == 0 and not obj_is_dict:
        raise Exception(f"Object {obj} not acceptable.")
    
    if obj_is_dict:
        iter_items = obj.items()
        response = {}
        
    elif obj_has_iter:
        iter_items = [(None, item) for item in obj]
        response = []
    
    for key, value in iter_items:
        new_key = str(key)
        
        if is_dict(value):
            new_val = convert_keys_to_strings(value, depth+1)
            
        elif has_iter(value):
            new_val = []
            
            for item in value:
                
                new_value = item
                
                if is_dict(item) or has_iter(item):
                    new_value = convert_keys_to_strings(item, depth+1)
                    
                new_val.append(new_value)
            
        else:
            new_val = value
            
        if obj_is_dict:
            response[new_key] = new_val
            
        elif obj_has_iter:
            response.append(new_val)
            
    return response


class CodeTypes(Enum):
    SUCCESS = "Success"
    IGNORED = "Ignored"
    NOT_EXPANDED = "Not expanded"
    ERROR = "Error"
    DEPTH_LIMIT = "Depth limit"


class ReasonTypes(Enum):
    EXPANDED_AT_OTHER_LOCATION = "Expanded at other location"
    IGNORE_ABS_NAME_SPACE = "Ignore absolute name space"
    IGNORE_ATTRIBUTE = "Ignore attribute"
    IGNORE_CLASS = "Ignore class"
    REASON_NOT_APPLICABLE = "Reason not applicable"
    ERROR = "Error when try expand"
    DEPTH_LIMIT_REACHED = "Depth limit reached"


class ClassInfo(TypedDict):
    nameSpace: str
    className: str
    code: CodeTypes
    reason: ReasonTypes
    error: Optional[str]


class ExpandedAtInfo(ClassInfo):
    expandedAt: str


class ExpandedAtMetaInfo(TypedDict):
    __meta__: ExpandedAtInfo


def to_dict(
    obj: Any,
    ignore_meta_when_expanded: bool=True,
    self_class_name: str="self",
    ignore_attr: Optional[List[str]]=None,
    ignore_abs_name_spaces: Optional[List[str]]=None,
    custom_treatment_namespaces: Optional[List[CustomTreatment]]=None,
    custom_treatment_types: Optional[List[CustomTreatment]]=None,
    custom_treatment_obj_is: Optional[List[CustomTreatment]]=None,
    custom_treatment_obj_eq: Optional[List[CustomTreatment]]=None,
    custom_treatment_conditional: Optional[List[CustomTreatment]]=None,
    ignore_exceptions: bool=False,
    exception_handler: Callable=None,
    depth_limit: int=5
) -> Dict:
    """Get Dict representation of instantiated object.

    Parameters
    ----------
    obj : Any
        The object to get value

    Returns
    -------
    Dict
        The Dict representation of object.
    """
    
    if ignore_attr is None:
        ignore_attr = []

    if ignore_abs_name_spaces is None:
        ignore_abs_name_spaces = []

    if custom_treatment_namespaces is None:
        custom_treatment_namespaces = []

    if custom_treatment_types is None:
        custom_treatment_types = []

    if custom_treatment_obj_is is None:
        custom_treatment_obj_is = []

    if custom_treatment_obj_eq is None:
        custom_treatment_obj_eq = []

    if custom_treatment_conditional is None:
        custom_treatment_conditional = []

    
    expanded_at: Dict[Any, ExpandedAtMetaInfo] = {}
    
    CUSTOM_TREATMENT_NAMESPACES = {
        custom_treat.fullname: custom_treat
        for custom_treat in custom_treatment_namespaces
    }
    CUSTOM_TREATMENT_TYPES = {
        custom_treat.fullname: custom_treat
        for custom_treat in custom_treatment_types
    }
    
    def _to_dict(obj: Any, name_space: str, depth: int=0) -> Dict:
        
        attr_name = name_space.split(".")[-1]
        
        try:
            
            ### Check Policies
            
            ################################################################
            # Depth Limit | depth-limit
            ################################################################
            if depth > depth_limit:
                # raise Exception(f"Limit depth was reach. Limit Depth: {depth_limit:,}")
                return {
                    '__meta__': ClassInfo(
                        nameSpace=name_space,
                        className=attr_name,
                        code=CodeTypes.DEPTH_LIMIT.name,
                        reason=ReasonTypes.DEPTH_LIMIT_REACHED.name,
                    )
                }
            
            ################################################################
            # Ignore abs name spaces | ignore-abs-name-spaces
            ################################################################
            if name_space in ignore_abs_name_spaces:
                resp = {}
                resp['__meta__'] = ClassInfo(
                        nameSpace=name_space,
                        className=attr_name,
                        code=CodeTypes.IGNORED.name,
                        reason=ReasonTypes.IGNORE_ABS_NAME_SPACE.name
                    )
                return resp
            
            ################################################################
            # Ignore attr | ignore-attr
            ################################################################
            if attr_name in ignore_attr:
                resp = {}
                resp['__meta__'] = ClassInfo(
                        nameSpace=name_space,
                        className=attr_name,
                        code=CodeTypes.IGNORED.name,
                        reason=ReasonTypes.IGNORE_ATTRIBUTE.name
                    )
                return resp
            
            ################################################################
            # Custom treatment namespaces | custom-treatment-namespaces
            ################################################################
            if name_space in CUSTOM_TREATMENT_NAMESPACES:
                custom_treat = CUSTOM_TREATMENT_NAMESPACES[name_space]
                return custom_treat.apply(obj)
            
            ################################################################
            # Custom treatment types | custom-treatment-types
            ################################################################
            if str(type(obj).__name__) in CUSTOM_TREATMENT_TYPES:
                custom_treat = CUSTOM_TREATMENT_TYPES[str(type(obj).__name__)]
                return custom_treat.apply(obj)
            
            ################################################################
            # Custom treatment when obj is | custom-treatment-when-obj-is
            ################################################################
            for custom_treat in custom_treatment_obj_is:
                if obj is custom_treat.fullname:
                    return custom_treat.apply(obj)
            
            ################################################################
            # Custom treatment when obj eq | custom-treatment-when-obj-eq
            ################################################################
            for custom_treat in custom_treatment_obj_eq:
                if obj == custom_treat.fullname:
                    return custom_treat.apply(obj)
            
            ################################################################
            # Custom treatment conditional | custom-treatment-conditional
            ################################################################
            for custom_treat in custom_treatment_conditional:
                if custom_treat.fullname(obj):
                    return custom_treat.apply(obj)
            
            
            
            ################################################################
            # Default Handle Obj
            ################################################################
            
            # If obj is instance of dict
            if isinstance(obj, dict):
                data = {}
                for (k, v) in obj.items():
                    item_name_space = f'{name_space}.{k}'
                    item_append = _to_dict(v, item_name_space, depth+1)
                    data[k] = item_append
                return data
            
            # If obj has _ast
            if hasattr(obj, "_ast"):
                return _to_dict(obj._ast(), name_space, depth+1)
            
            
            # If obj has __iter__
            if hasattr(obj, "__iter__") and not isinstance(obj, str):       
                idx = 0
                list_resp = []
                for v in obj:
                    item_name_space = f'{name_space}[{idx}]'
                    item_append = _to_dict(v, item_name_space, depth+1)
                    list_resp.append(item_append)
                    idx = idx + 1
                        
                return list_resp
            
            # If obj has __dict__
            if hasattr(obj, "__dict__"):
                
                if expanded_at.get(obj, None) is not None:
                    return expanded_at[obj]
                
                else:
                    class_name = None
                    if hasattr(obj, "__class__"):
                        class_name = obj.__class__.__name__
                        
                    expanded_at[obj] = ExpandedAtMetaInfo(
                        __meta__=ExpandedAtInfo(
                            nameSpace=None,
                            className=class_name,
                            expandedAt=name_space,
                            code=CodeTypes.NOT_EXPANDED.name,
                            reason=ReasonTypes.EXPANDED_AT_OTHER_LOCATION.name
                        )
                    )
                    
                    attr_name = name_space.split(".")[-1]

                    get_info_from_keys = list(obj.__dict__.keys())
                    for key_dir in obj.__dir__():
                        if key_dir not in get_info_from_keys \
                        and not key_dir.startswith('__') \
                        and not key_dir.endswith('__') \
                        and not callable(getattr(obj, key_dir)) \
                        and getattr(obj, key_dir) is not None:
                            get_info_from_keys.append(key_dir)
                    
                    data = dict([
                            (
                                key, 
                                _to_dict(getattr(obj, key), f'{name_space}.{key}', depth+1)
                            ) 
                            for key in get_info_from_keys
                        ])
                    
                    if not ignore_meta_when_expanded:
                        data["__meta__"]: ClassInfo = ClassInfo(
                                nameSpace=name_space,
                                className=class_name,
                                code=CodeTypes.SUCCESS.name,
                                reason=ReasonTypes.REASON_NOT_APPLICABLE.name
                            )
                        
                    return data
            
            # Check None
            if obj is None:
                return obj
            
            # Check natural types
            for natural_type in [str, int, float, bool]:
                if isinstance(obj, natural_type):
                    return obj
            
            # Return string of obj
            return str(obj)
            
        except Exception as exc:
            if not ignore_exceptions:
                raise exc
            if exception_handler is None:
                return {
                    '__meta__': ClassInfo(
                        nameSpace=name_space,
                        className=attr_name,
                        code=CodeTypes.ERROR.name,
                        reason=ReasonTypes.ERROR.name,
                        error=str(exc)
                    )
                }
            return exception_handler(exc)
    
    resp_dict = _to_dict(obj, self_class_name)
    resp_dict = convert_keys_to_strings(resp_dict)
    return resp_dict

    
def get_dict_val(
    data: Dict,
    keys_depth: List[str],
    default_val: Any=None
) -> Any:
    """Try get data from list of keys

    Parameters
    ----------
    data : Dict
        Dict data
    keys_depth : List[str]
        List of keys to find data
    default_val : _type_, optional
        If not find, return this value, by default None

    Returns
    -------
    Any
        Value from Dict
    """
    curr_data = None
    for idx, k in enumerate(keys_depth):
        if idx == 0:
            d = data.get(k, None)
        else:
            d = curr_data.get(k, None)
        if d is None:
            return default_val
        if idx < len(keys_depth) - 1:
            curr_data = deepcopy(d)
    return d


def split_list(
    input_list: List, 
    batch_size: int
) -> List[List]:
    sub_lists = [
        input_list[i: i+batch_size] 
        for i in range(0, len(input_list), batch_size)
    ]
    return sub_lists


def split_into_chunks(
    input_list: List, 
    chunksize: int=1
):
    if len(input_list) == 0 \
    or chunksize == 0:
        raise ValueError("List cannot be empty and group lenght must be greater than zero.")

    chunks_list = []
    batch_size: int = len(input_list) // chunksize

    if batch_size > 0:
        for i in range(0, len(input_list), batch_size):
            chunk = input_list[i:i + batch_size]
            chunks_list.append(chunk)
    
    else:
        chunks_list = [input_list]

    return chunks_list


def iso_parser(date_str: str) -> datetime:
    """Iso Parser.
    """
    response = None
    error = False
    try:
        response = datetime.fromisoformat(date_str)
    except:
        error = True    
    if error:
        error = False
        try:
            response = parser.isoparse(date_str)
        except:
            error = True            
    if error:
        raise Exception(f"Invalid date: {date_str}")
    return response


def smart_tz_handle(date_str):
    """Parse date as string handling timezone.
    """
    try:
        date = parser.parse(date_str)
        if date.tzinfo is None:
            date = date.replace(tzinfo=tz.tzutc())
        # Retorna a representação em string da data com o fuso horário UTC
        return date.strftime('%Y-%m-%dT%H:%M:%S.%f %z')
    except Exception as e:
        # print(f"Erro ao processar a data: {e}")
        return None
    
    
class ExpectedRemainingTimeHandle:
    """Handle expected remaining time on iterations or 
    thread executions.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.started_at = datetime.now()

    def _total_seconds(self, executed_num: int):
        return (datetime.now() - self.started_at).total_seconds()

    def _speed(self, executed_num: int):
        return self._total_seconds(executed_num) / executed_num

    def seconds(self, executed_num: int) -> float:
        speed = self._speed(executed_num)
        return speed * (self.total - executed_num)

    def display_time(self, executed_num: int, granularity: int=2) -> str:
        return display_time(math.ceil(self.seconds(executed_num)), granularity=granularity)