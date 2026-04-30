

from typing import List, Dict
import pandas as pd
from drtools.utils import list_ops
from ...custom.assets.load_assets import TabularAsMatrixLoadAsset
from .resource import BaseGsheetsResource
from .source import GoogleSheetsSource


class GoogleSheetsUpdateOrInsertLoadAsset(TabularAsMatrixLoadAsset):
    """LoadAsset that performs an upsert (update-or-insert) operation on a Google Sheets worksheet.

    Compares incoming data against the current sheet content using UNIQUE_KEYS to build a
    composite row ID. Rows present in the sheet but absent from the new data are kept;
    rows present in the new data replace or extend the existing ones. The final result is
    sorted by 'created_at' and written back to the sheet via SOURCE.push().

    Class attributes:
        UNIQUE_KEYS: List of column names used to build the composite row identifier.
            Must be set by subclasses.
        SOURCE: GoogleSheetsSource subclass defining the target sheet and schema.
            Must be set by subclasses.
        RESOURCES: Must contain exactly one BaseGsheetsResource subclass.

    Raises:
        Exception: On init if SOURCE is not a GoogleSheetsSource, RESOURCES is invalid,
            or UNIQUE_KEYS is not set.
    """

    UNIQUE_KEYS: List[str] = None
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

    def load(self, data: List[List[str]]) -> Dict:
        """Upsert data into the target Google Sheets worksheet.

        Steps:
        1. Convert incoming matrix to DataFrame and compute composite row IDs.
        2. Fetch current sheet content via SOURCE.fetch().
        3. Keep rows that exist in the sheet but are absent from the new data.
        4. Apply auto-column values to kept and new rows.
        5. Concatenate, sort by 'created_at', and write back via SOURCE.push().

        Args:
            data: Matrix (list of lists) where the first row is the header.

        Returns:
            The gspread update response dict.
        """
        resources = self.get_instantiated_resources_list()
        all_columns = self.SOURCE.list_all_column_names()

        def _construct_id(row):
            return ";".join(str(row[col]) for col in self.UNIQUE_KEYS)

        new_data = pd.DataFrame(columns=data[0], data=data[1:])
        new_data['_id'] = new_data.apply(_construct_id, axis=1)
        new_data = new_data.set_index('_id')
        for col in self.SOURCE.list_auto_columns():
            new_data[col.name] = col.auto_value()
        new_data = new_data[all_columns]

        current_df = self.SOURCE.fetch(resources=resources)
        self.LOGGER.debug(f'Current sheet shape: {current_df.shape}')

        if current_df.shape[0] > 0:
            current_df['_id'] = current_df.apply(_construct_id, axis=1)

            ids_only_on_sheet = list_ops(
                current_df['_id'].unique().tolist(),
                new_data.index.tolist(),
            )
            keep_data = current_df[current_df['_id'].isin(ids_only_on_sheet)].set_index('_id')
            for col in self.SOURCE.list_auto_columns():
                if not col.auto_add:
                    keep_data[col.name] = col.auto_value()
            keep_data = keep_data[all_columns]

            self.LOGGER.debug(f'Keep data shape: {keep_data.shape}')
            insert_df = pd.concat([keep_data, new_data], axis=0)
        else:
            insert_df = new_data

        insert_df = insert_df.sort_values(['created_at']).reset_index(drop=True)
        self.LOGGER.debug(f'Insert data shape: {insert_df.shape}')

        return self.SOURCE.push(insert_df, resources=resources)
