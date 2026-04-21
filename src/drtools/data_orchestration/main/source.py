

from .common import ContextComponent
from typing import List
from enum import Enum
from datetime import datetime
from drtools.utils import list_ops
from .utils import has_unique_elements, get_duplicates


class ColumnType(Enum):
    """Enumeration of supported column data types."""

    STRING = ("string",)
    DOUBLE = ("double",)
    INTEGER = ("integer",)
    DATE = ("date",) # YYYY-MM-DD
    DATETIME = ("datetime",)

    @property
    def type_name(self) -> str:
        """Return the string identifier of the column type (e.g. 'string', 'integer')."""
        return self.value[0]


class Column:
    """Base class representing a schema column with type and nullability rules.

    Attributes:
        name: Column identifier.
        type: ColumnType value for this column.
        auto: If True, value is generated automatically on every write.
        auto_add: If True, value is generated only on insert (not on update).
        null: If True, None values are accepted.
        blank: If True, empty string values are accepted.
    """

    def __init__(
        self,
        name: str,
        type: str,
        auto: bool=False,
        auto_add: bool=False,
        null: bool=False,
        blank: bool=False,
    ):
        self.name = name
        self.type = type
        self.auto = auto
        self.auto_add = auto_add
        self.null = null
        self.blank = blank

    @property
    def is_auto(self) -> bool:
        """Return True if this column's value is automatically generated (auto or auto_add)."""
        return self.auto or self.auto_add

    def auto_value(self):
        """Return the auto-generated value for this column. Must be implemented by subclasses."""
        raise NotImplementedError

    def apply_type(self, value):
        """Cast value to this column's type. Must be implemented by subclasses."""
        raise NotImplementedError

    def validate_value(self, value):
        """Validate value against null and blank constraints.

        Raises:
            Exception: If value is None and null=False, or empty string and blank=False.
        """
        if value is None and not self.null:
            raise Exception(f"Column {self.name} can not be null.")
        if value == "" and not self.blank:
            raise Exception(f"Column {self.name} can not be blank.")
    

class StringColumn(Column):
    """Column that stores string values."""

    def __init__(self, *args, **kwargs):
        kwargs['type'] = ColumnType.STRING
        super().__init__(*args, **kwargs)

    def apply_type(self, value):
        if value is None:
            return value
        return str(value)

    def validate_value(self, value):
        """Validate and ensure value is castable to str.

        Raises:
            Exception: If value cannot be cast to string.
        """
        super().validate_value(value)
        if value is None or value == "":
            return
        try:
            self.apply_type(value)
        except Exception as exc:
            raise Exception(f"Invalid value {value} for column {self.name}. Expected string.")
        # if not isinstance(value, str):
        #     raise Exception(f"Invalid value {value} for column {self.name}. Expected string.")
        
        
class DoubleColumn(Column):
    """Column that stores floating-point values."""

    def __init__(self, *args, **kwargs):
        kwargs['type'] = ColumnType.DOUBLE
        super().__init__(*args, **kwargs)

    def apply_type(self, value):
        if value is None:
            return value
        return float(value)

    def validate_value(self, value):
        """Validate and ensure value is castable to float.

        Raises:
            Exception: If value cannot be cast to float.
        """
        super().validate_value(value)
        if value is None:
            return
        try:
            self.apply_type(value)
        except Exception as exc:
            raise Exception(f"Invalid value {value} for column {self.name}. Expected double.")
        # if not isinstance(value, float):
        #     raise Exception(f"Invalid value {value} for column {self.name}. Expected double.")
        
        
class IntegerColumn(Column):
    """Column that stores integer values."""

    def __init__(self, *args, **kwargs):
        kwargs['type'] = ColumnType.INTEGER
        super().__init__(*args, **kwargs)

    def apply_type(self, value):
        if value is None:
            return value
        return int(value)

    def validate_value(self, value):
        """Validate and ensure value is castable to int.

        Raises:
            Exception: If value cannot be cast to int.
        """
        super().validate_value(value)
        if value is None:
            return
        try:
            self.apply_type(value)
        except Exception as exc:
            raise Exception(f"Invalid value {value} for column {self.name}. Expected integer.")
        # if not isinstance(value, int):
        #     raise Exception(f"Invalid value {value} for column {self.name}. Expected integer.")
        
        
class DateColumn(Column):
    """Column that stores dates in YYYY-MM-DD format."""

    def __init__(self, *args, **kwargs):
        kwargs['type'] = ColumnType.DATE
        super().__init__(*args, **kwargs)

    def apply_type(self, value):
        if value is None:
            return value
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")

    def validate_value(self, value):
        """Validate that value matches the YYYY-MM-DD format.

        Raises:
            Exception: If value is not a valid YYYY-MM-DD date string.
        """
        super().validate_value(value)
        if value is None:
            return
        invalid_msg = f"Invalid value {value} for column {self.name}. Expected date format YYYY-MM-DD."
        try:
            self.apply_type(value)
        except Exception as exc:
            raise Exception(invalid_msg)
        
        
class DatetimeColumn(Column):
    """Column that stores datetimes in ISO 8601 format."""

    def __init__(self, *args, **kwargs):
        kwargs['type'] = ColumnType.DATETIME
        super().__init__(*args, **kwargs)

    def apply_type(self, value):
        if value is None:
            return value
        return datetime.fromisoformat(value).isoformat()

    def validate_value(self, value):
        """Validate that value is a valid ISO 8601 datetime string.

        Raises:
            Exception: If value cannot be parsed as an ISO datetime.
        """
        super().validate_value(value)
        if value is None:
            return
        invalid_msg = f"Invalid value {value} for column {self.name}. Expected datetime isoformat YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS.ss"
        try:
            self.apply_type(value)
        except Exception as exc:
            raise Exception(invalid_msg)
        # if not isinstance(value, str):
        #     raise Exception(invalid_msg)
        # try:
        #     datetime.fromisoformat(value)
        # except ValueError:
        #     raise Exception(invalid_msg)

        
# class AutoDatetimeColumn(DatetimeColumn):
#     def __init__(self, *args, **kwargs):
#         kwargs['auto'] = True
#         super().__init__(*args, **kwargs)
        
#     def auto_value(self):
#         return datetime.now().isoformat()


class Source(ContextComponent):
    """Base class for all data sources."""

    NAME: str = None
    DESCRIPTION: str = None
    ALIAS: str = None


class TabularSource(Source):
    """Source with a defined column schema, supporting record validation and type coercion.

    Class attributes:
        COLUMNS: List of Column instances defining the schema. Must be set by subclasses.

    Raises:
        Exception: On init if COLUMNS is not set or contains duplicate column names.
    """

    COLUMNS: List[Column] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.COLUMNS:
            raise Exception("Static attribute COLUMNS must be set.")
        self._column_names = [col.name for col in self.COLUMNS]
        if not has_unique_elements(self._column_names):
            duplicated_column_names = get_duplicates(self._column_names)
            raise Exception(f"Columns names must be unique. Found duplicated: {duplicated_column_names}")
    
    def list_manual_columns(self) -> List[str]:
        """Return Column instances for all non-auto columns."""
        return [col for col in self.COLUMNS if not col.is_auto]

    def list_auto_columns(self) -> List[str]:
        """Return Column instances for all auto-generated columns."""
        return [col for col in self.COLUMNS if col.is_auto]

    def list_all_columns(self) -> List[str]:
        """Return all Column instances."""
        return [col for col in self.COLUMNS]

    def list_manual_column_names(self) -> List[str]:
        """Return names of all non-auto columns."""
        return [col.name for col in self.COLUMNS if not col.is_auto]

    def list_auto_column_names(self) -> List[str]:
        """Return names of all auto-generated columns."""
        return [col.name for col in self.COLUMNS if col.is_auto]

    def list_all_column_names(self) -> List[str]:
        """Return names of all columns."""
        return [col.name for col in self.COLUMNS]

    def get_column_by_name(self, name: str) -> Column:
        """Return the Column instance with the given name.

        Raises:
            Exception: If no column with that name exists.
        """
        found_col = [col for col in self.COLUMNS if col.name == name]
        if not found_col:
            raise Exception(f"Column with name {name} was not found on source.")
        if len(found_col) > 1:
            raise Exception(f"Has more than 1 column with name {name} on source.")
        return found_col[0]

    def get_column_names_from_level(
        self,
        level: str="manual" # all, manual, auto
    ) -> List[str]:
        """Return column names filtered by level: 'all', 'manual', or 'auto'.

        Raises:
            Exception: If level is not one of the valid options.
        """
        if level == "manual":
            columns = self.list_manual_column_names()
        elif level == "all":
            columns = self.list_all_column_names()
        elif level == "auto":
            columns = self.list_auto_column_names()
        else:
            raise Exception(f"Invalid receveid columns level {level}. Valid are 'all', 'manual' and 'auto'.")
        return columns
        
    def validate_record_as_dict(
        self,
        record_as_dict,
        columns_level: str="manual" # all, manual, auto
    ):
        """Validate a single record (dict) against the schema.

        Checks that keys match the expected columns and each value passes its column's validation.

        Raises:
            Exception: If keys mismatch or any value fails validation.
        """
        columns = self.get_column_names_from_level(columns_level)
        keys_list = list(record_as_dict.keys())
        if sorted(columns) != sorted(keys_list):
            unexpected_keys = list_ops(columns, keys_list)
            if not unexpected_keys:
                unexpected_keys = list_ops(keys_list, columns)
            raise Exception(f"Expected keys {columns} on record, following columns are missing or are unexpected: {unexpected_keys}.")
        for column_name, value in record_as_dict.items():
            column = self.get_column_by_name(column_name)
            column.validate_value(value)
    
    def validate_list_of_records_as_dict(
        self,
        list_of_records_as_dict,
        columns_level: str="manual" # all, manual, auto
    ):
        """Validate a list of dict records against the schema."""
        for record_as_dict in list_of_records_as_dict:
            self.validate_record_as_dict(record_as_dict, columns_level)
    
    def validate_records_as_dataframe(
        self,
        records_as_dataframe,
        columns_level: str="manual" # all, manual, auto
    ):
        """Validate all rows of a DataFrame against the schema."""
        list_of_records_as_dict = records_as_dataframe.to_dict(orient="records")
        for record_as_dict in list_of_records_as_dict:
            self.validate_record_as_dict(record_as_dict, columns_level)
    
    def validate_record_as_list(
        self,
        record_as_list,
        columns_level: str="manual" # all, manual, auto
    ):
        """Validate a single record (list) against the schema by positional column mapping.

        Raises:
            Exception: If the list length does not match expected columns, or any value fails.
        """
        columns = self.get_column_names_from_level(columns_level)
        expected_len = len(columns)
        received_len = len(record_as_list)
        if expected_len != received_len:
            raise Exception(f"Expected list with {expected_len} items. Received {received_len} items.")
        for idx, value in enumerate(record_as_list):
            column = self.get_column_by_name(columns[idx])
            column.validate_value(value)
    
    def validate_list_of_records_as_list(
        self,
        list_of_records_as_list,
        columns_level: str="manual" # all, manual, auto
    ):
        """Validate a list of list-records against the schema."""
        for record_as_list in list_of_records_as_list:
            self.validate_record_as_list(record_as_list, columns_level)
            
    def type_record_as_dict(
        self,
        record_as_dict,
        columns_level: str="manual" # all, manual, auto
    ):
        """Validate and cast each value in a dict record to its column type.

        Returns:
            A new dict with type-coerced values.
        """
        self.validate_record_as_dict(record_as_dict, columns_level)
        typed_record = {}
        for column_name, value in record_as_dict.items():
            typed_record[column_name] = self.get_column_by_name(column_name).apply_type(value)
        return typed_record
            
    def type_record_as_list(self, record_as_list, columns_level: str="manual"):
        """Validate and cast each value in a list record to its column type.

        Returns:
            A new list with type-coerced values.
        """
        self.validate_record_as_list(record_as_list, columns_level)
        columns = self.get_column_names_from_level(columns_level)
        typed_record = []
        for column_name, value in zip(columns, record_as_list):
            typed_record.append(self.get_column_by_name(column_name).apply_type(value))
        return typed_record
    
    def type_list_of_records_as_dict(
        self,
        list_of_records_as_dict,
        columns_level: str="manual" # all, manual, auto
    ):
        """Validate and cast all dict records in a list."""
        self.validate_list_of_records_as_dict(list_of_records_as_dict, columns_level)
        typed_list_of_records_as_dict = []
        for record_as_dict in list_of_records_as_dict:
            typed_list_of_records_as_dict.append(self.type_record_as_dict(record_as_dict))
        return typed_list_of_records_as_dict
    
    def type_list_of_records_as_list(
        self,
        list_of_records_as_list,
        columns_level: str="manual" # all, manual, auto
    ):
        """Validate and cast all list records in a list."""
        self.validate_list_of_records_as_list(list_of_records_as_list, columns_level)
        typed_list_of_records_as_list = []
        for record_as_list in list_of_records_as_list:
            typed_list_of_records_as_list.append(self.type_record_as_list(record_as_list))
        return typed_list_of_records_as_list