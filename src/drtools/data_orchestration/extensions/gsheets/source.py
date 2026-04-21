


from ...main.source import TabularSource


class GoogleSheetsSource(TabularSource):
    """TabularSource backed by a specific Google Sheets worksheet.

    Class attributes:
        GSHEETS_ID: Google Sheets document ID. Must be set by subclasses.
        SHEET: Worksheet tab name. Must be set by subclasses.

    Raises:
        Exception: On init if GSHEETS_ID or SHEET are not set.
    """

    GSHEETS_ID = None
    SHEET = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.GSHEETS_ID:
            raise Exception("Static attribute GSHEETS_ID must be set.")
        if not self.SHEET:
            raise Exception("Static attribute SHEET must be set.")