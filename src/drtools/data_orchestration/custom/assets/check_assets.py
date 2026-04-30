

from ...main.assets import CheckAsset, AssetCheckResult
from pandas import DataFrame
from typing import List


class CheckDataIsDataframe(CheckAsset):
    """Check that the asset data is a pandas DataFrame."""

    NAME: str = "CheckDataIsDataframe"
    DESCRIPTION: str = "Check if Data is DataFrame"
    
    def check(self, asset_data: DataFrame):
        if not isinstance(asset_data, DataFrame):
            return AssetCheckResult(passed=False)
        return AssetCheckResult(passed=True)


class CheckDataIsMatrix(CheckAsset):
    """Check that the asset data is a list of lists (matrix format)."""

    NAME: str = "CheckDataIsMatrix"
    DESCRIPTION: str = "Check if Data is matrix"
    
    def check(self, asset_data: List[List]):
        if not isinstance(asset_data, list):
            return AssetCheckResult(passed=False, metadata={'message': "Asset Data is not matrix."})
        for idx, asset_data_row in enumerate(asset_data):
            if not isinstance(asset_data_row, list):
                return AssetCheckResult(passed=False, metadata={'message': f"Row with {idx} from Asset Data is not list."})
        return AssetCheckResult(passed=True)


class CheckAllColumnsFromDataframe(CheckAsset):
    """Check that the DataFrame columns match all columns defined in the asset's source schema."""

    NAME: str = "CheckAllColumnsFromDataframe"
    DESCRIPTION: str = "Check headers of DataFrame Data"
    
    def check(self, asset_data: DataFrame):
        if self.asset.SOURCE.list_all_column_names() != asset_data.columns.tolist():
            return AssetCheckResult(passed=False)
        return AssetCheckResult(passed=True)
    

class CheckManualColumnsFromMatrixData(CheckAsset):
    """Check that the first row (header) of a matrix matches the manual columns of the asset's source."""

    NAME: str = "CheckManualColumnsFromMatrixData"
    DESCRIPTION: str = "Check headers of matrix"
    
    def check(self, asset_data: DataFrame):
        if self.asset.SOURCE.list_manual_column_names() != asset_data[0]:
            return AssetCheckResult(passed=False)
        return AssetCheckResult(passed=True)
    

class CheckAllColumnsFromMatrix(CheckAsset):
    """Check that the first row (header) of a matrix matches all columns of the asset's source."""

    NAME: str = "CheckAllColumnsFromMatrix"
    DESCRIPTION: str = "Check headers of matrix"
    
    def check(self, asset_data: DataFrame):
        if self.asset.SOURCE.list_all_column_names() != asset_data[0]:
            return AssetCheckResult(passed=False)
        return AssetCheckResult(passed=True)
    

class ValidateDataFromMatrix(CheckAsset):
    """Validate all data rows in a matrix against the asset's source schema."""

    NAME: str = "CheckDataFromMatrix"
    def check(self, asset_data: DataFrame):
        passed = False
        metadata = {}
        try:
            self.asset.SOURCE.validate_list_of_records_as_list(asset_data)
            passed = True
        except Exception as exc:
            metadata['message'] = str(exc)
            passed = False
        return AssetCheckResult(passed=passed, metadata=metadata)
    

class ValidateDataFromDataframe(CheckAsset):
    """Validate all rows in a DataFrame against the asset's source schema."""

    NAME: str = "ValidateDataFromDataframe"
    def check(self, asset_data: DataFrame):
        passed = False
        metadata = {}
        try:
            self.asset.SOURCE.validate_records_as_dataframe(asset_data)
            passed = True
        except Exception as exc:
            metadata['message'] = str(exc)
            passed = False
        return AssetCheckResult(passed=passed, metadata=metadata)