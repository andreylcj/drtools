

from typing import TypedDict, Tuple, Dict


class ArgsKwargs(TypedDict):
    """Container for positional and keyword arguments to be passed between pipeline stages.

    Attributes:
        args: Positional arguments tuple.
        kwargs: Keyword arguments dict.
    """

    args: Tuple
    kwargs: Dict