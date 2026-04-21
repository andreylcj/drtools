

from typing import List, Dict
import pandas as pd
from drtools.utils import list_ops
from ...custom.assets.load_assets import TabularAsMatrixLoadAsset
from .resource import BaseGsheetsResource
from .source import GoogleSheetsSource
from .data_asset import GoogleSheetsDataAsset


class GoogleSheetsUpdateOrInsertLoadAsset(TabularAsMatrixLoadAsset):
    """LoadAsset that performs an upsert (update-or-insert) operation on a Google Sheets worksheet.

    Compares incoming data against the current sheet content using UNIQUE_KEYS to build a
    composite row ID. Rows present in the sheet but absent from the new data are kept;
    rows present in the new data replace or extend the existing ones. The final result is
    sorted and written back to the sheet.

    Class attributes:
        UNIQUE_KEYS: List of column names used to build the composite row identifier.
            Must be set by subclasses.
        SOURCE: GoogleSheetsSource class defining the target sheet and schema.
            Must be set by subclasses.
        RESOURCES: Must contain exactly one BaseGsheetsResource.

    Raises:
        Exception: On init if SOURCE is not a GoogleSheetsSource, RESOURCES is invalid,
            UNIQUE_KEYS is not set, or SOURCE_DATA_ASSET is not a GoogleSheetsDataAsset.
    """

    UNIQUE_KEYS: List[str] = None
    # SOURCE_DATA_ASSET = None
    SOURCE = None
    RESOURCES = [BaseGsheetsResource]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.SOURCE, GoogleSheetsSource):
            raise Exception("Source must be an instance of extensions.gsheets.source.GoogleSheetsSource.")
        if not self.RESOURCES:
            raise Exception("Static attribute RESOURCES must be set.")
        if len(self.RESOURCES) > 1:
            raise Exception("Static attribute RESOURCES must have just one Google Sheets Resource.")
        if not isinstance(self.get_resource(self.RESOURCES[0].ALIAS), BaseGsheetsResource):
            raise Exception("Resource must be instance of extensions.gsheets.resource.BaseGsheetsResource.")
        if not self.UNIQUE_KEYS:
            raise Exception("Static attribute UNIQUE_KEYS must be set.")
        # if not self.SOURCE_DATA_ASSET:
        #     raise Exception("Static attribute SOURCE_DATA_ASSET must be set.")
        # self.SOURCE_DATA_ASSET = self.SOURCE_DATA_ASSET(conf=self.context.conf, LOGGER=self.LOGGER)
        # if not isinstance(self.SOURCE_DATA_ASSET, GoogleSheetsDataAsset):
        #     raise Exception("Source Data Asset must be an instance of extensions.gsheets.data_asset.GoogleSheetsDataAsset.")
        self.GSHEETS_ID = self.SOURCE.GSHEETS_ID
        self.SHEET = self.SOURCE.SHEET
        self.source_data = None
    
    def get_source_data(self):
        """Return the current sheet content as a DataFrame, materializing it on first call."""
        if not self.source_data:
            self.source_data = self.SOURCE.materialize()
        return self.source_data
    
    def load(self, data: List[List[str]]) -> Dict:
        """Upsert data into the target Google Sheets worksheet.

        Steps:
        1. Convert incoming matrix to DataFrame and compute composite row IDs.
        2. Fetch current sheet content via SOURCE.
        3. Keep rows that exist in the sheet but are absent from the new data.
        4. Apply auto-column values to kept and new rows.
        5. Concatenate, sort by 'created_at', and write back to the sheet.

        Args:
            data: Matrix (list of lists) where the first row is the header.

        Returns:
            The gspread update response dict.
        """
        data = pd.DataFrame(columns=data[0], data=data[1:])
        
        def _construct_id(row):
            _id = ""
            for col in self.UNIQUE_KEYS:
                _id += row[col] + ";"
            _id = _id[:-1]
            return _id
        data['_id'] = data.apply(_construct_id, axis=1)
        
        # Load sheet as df
        raw_current_data_df = self.get_source_data()
        empty_sheet = True
        if raw_current_data_df.shape[0] > 0:
            empty_sheet = False
        if not empty_sheet:
            raw_current_data_df['_id'] = raw_current_data_df.apply(_construct_id, axis=1)
            curr_data_df = raw_current_data_df[self.SOURCE.list_all_column_names()]
        curr_shape = raw_current_data_df.shape
        if curr_data_df:
            curr_shape = curr_data_df.shape
        self.LOGGER.debug(f'Sheet DataFrame Shape: {curr_shape}')
        self.LOGGER.debug('Load sheet as DataFrame... Done!')
        
        # Compute insert data
        new_data = data.set_index('_id')
        for col in self.SOURCE.list_auto_columns():
            new_data[col.name] = col.auto_value()
        new_data = new_data[self.SOURCE.list_all_column_names()]
        
        if not empty_sheet:
            code_data_on_sh_but_not_on_extraction = list_ops(curr_data_df._id.unique(), data._id.unique())
            keep_data = curr_data_df[curr_data_df._id.isin(code_data_on_sh_but_not_on_extraction)]
            keep_data = keep_data.set_index('_id')
            self.LOGGER.debug(f"Keep Data shape: {keep_data.shape}")
            for col in self.SOURCE.list_auto_columns():
                if not col.auto_add:
                    keep_data[col.name] = col.auto_value()
            keep_data = keep_data[self.SOURCE.list_all_column_names()]
            insert_data = pd.concat([keep_data, new_data], axis=0)
        else:
            insert_data = data
            
        insert_data = insert_data[self.SOURCE.list_all_column_names()]
        
        self.LOGGER.debug(f"New Data shape: {new_data.shape}")
        self.LOGGER.debug(f"Insert Data shape: {insert_data.shape}")
        
        insert_data = insert_data.sort_values(['created_at'])
        insert_data = insert_data.reset_index(drop=True)
        insert_data = insert_data.fillna('').astype(str)
        insert_data = insert_data.values.tolist()
        self.insert_data = insert_data
        self.LOGGER.debug('Compute insert data... Done!')

        gsheets_resource = self.RESOURCES[0].ALIAS

        # Insert new data
        # update_res = gsheets_resource.update_sheet(
        #     insert_data, 
        #     self.GSHEETS_ID,
        #     self.SHEET, 
        #     'B2',
        #     raw=False
        # )
        
        # self.LOGGER.debug(update_res)
        # self.LOGGER.debug('Insert new data... Done!')
        
        # return update_res