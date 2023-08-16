

# from pandas import DataFrame, Series
# import pandas as pd
# from typing import List, Union, Dict, TypedDict, Any
# from drtools.logging import Logger, FormatterOptions


# class BaseExtractor:
#     pass


# class BaseTransformer:
#     def __init__(
#         self,
#         model=None, # drtools.data_science.model_handling.Model
#         LOGGER: Logger=Logger(
#             name="BaseTransformer",
#             formatter_options=FormatterOptions(
#                 include_datetime=True,
#                 include_logger_name=True,
#                 include_level_name=True,
#             ),
#             default_start=False
#         )
#     ) -> None:
#         self.model = model
#         self.LOGGER = LOGGER
    
#     def transform(self, *args, **kwargs) -> Any:
#         raise Exception("Must be implemented.")
    

# class BaseLoader:
#     pass
