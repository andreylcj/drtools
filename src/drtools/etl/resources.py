

from typing import List, Any


class DefaultAssignReceivedValues:
    
    def __init__(
        self,
        values: List[Any]=None
    ):
        if values is None:
            values = []
        for value in values:
            setattr(self, value.__name__, value)