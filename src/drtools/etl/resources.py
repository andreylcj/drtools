

from typing import List, Any


class DefaultAssignReceivedValues:
    """Utilitário que registra dinamicamente uma lista de objetos como atributos da instância.

    Cada item da lista é atribuído como atributo usando ``item.__name__`` como chave,
    o que é útil para registrar classes de extractors, transformers e loaders nas
    classes container ``From``, ``Through`` e ``To``.

    Args:
        values: Lista de objetos (tipicamente classes) a serem registrados como atributos.
                Se ``None``, usa lista vazia.

    Example:
        >>> class MyExtractor:
        ...     pass
        >>> container = DefaultAssignReceivedValues([MyExtractor])
        >>> hasattr(container, "MyExtractor")
        True
        >>> container.MyExtractor is MyExtractor
        True
    """

    def __init__(
        self,
        values: List[Any]=None
    ):
        if values is None:
            values = []
        for value in values:
            setattr(self, value.__name__, value)
