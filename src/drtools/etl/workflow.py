

from typing import Any, Callable
from drtools.logging import Logger, FormatterOptions
from drtools.etl.types import Date
from drtools.etl.etl import To, Through, From
from datetime import datetime, timedelta


def get_relative_date_method(diff: int) -> Date:
    """Retorna uma data relativa à data atual no formato ``"%Y-%m-%d"``.

    Args:
        diff: Número de dias em relação a hoje. Use valores negativos para
              datas no passado e positivos para datas no futuro.

    Returns:
        Data formatada como string ``"%Y-%m-%d"``.

    Example:
        >>> from drtools.etl.workflow import get_relative_date_method
        >>> get_relative_date_method(0)   # hoje
        '2026-03-20'
        >>> get_relative_date_method(-1)  # ontem
        '2026-03-19'
        >>> get_relative_date_method(7)   # daqui a uma semana
        '2026-03-27'
    """
    date = datetime.now() + timedelta(diff)
    date: Date = date.strftime("%Y-%m-%d")
    return date


class Workflow:
    """Classe base para todos os workflows da drtools.

    Fornece infraestrutura comum: logger configurado, nome do workflow e
    um callable para calcular datas relativas.

    Args:
        get_relative_date: Callable que recebe um inteiro (diff em dias) e
            retorna uma :class:`~drtools.etl.types.Date`. Padrão:
            :func:`get_relative_date_method`.
        LOGGER: Logger da drtools. Usa logger padrão ``"Workflow"`` se não informado.

    Attributes:
        name (str): Retorna :attr:`NAME` se definido, caso contrário o nome da classe.

    Example:
        >>> class DailySync(Workflow):
        ...     NAME = "daily-sync"
        ...
        ...     def run(self):
        ...         today = self.get_relative_date(0)
        ...         yesterday = self.get_relative_date(-1)
        ...         self.LOGGER.info(f"Syncing {yesterday} → {today}")
        ...
        >>> wf = DailySync()
        >>> wf.name
        'daily-sync'
    """

    NAME: str = None
    """Nome opcional do workflow. Se ``None``, usa o nome da classe."""

    def __init__(
        self,
        get_relative_date: Callable=get_relative_date_method,
        LOGGER: Logger=Logger(
            name="Workflow",
            formatter_options=FormatterOptions(
                include_datetime=True,
                include_thread_name=True,
                include_logger_name=True,
                include_level_name=True,
            ),
            default_start=False
        )
    ) -> None:
        self.get_relative_date = get_relative_date
        self.LOGGER = LOGGER

    @property
    def name(self) -> str:
        """Nome do workflow: retorna :attr:`NAME` se definido, senão o nome da classe."""
        return self.NAME or self.__class__.__name__


class ToThroughFromETLWorkflow(Workflow):
    """Workflow ETL declarativo com as três etapas clássicas: Extract → Transform → Load.

    Cada etapa é definida como um método abstrato (:meth:`extract`, :meth:`transform`,
    :meth:`load`) e orquestrada pelo método :meth:`run`.

    Os containers :attr:`From`, :attr:`Through` e :attr:`To` devem ser definidos como
    atributos estáticos na subclasse, tornando explícitas as dependências do workflow.

    Class Attributes:
        To (To): Container de loaders. **Obrigatório.**
        Through (Through): Container de transformers. **Obrigatório.**
        From (From): Container de extractors. **Obrigatório.**

    Raises:
        AssertionError: Se ``To``, ``Through`` ou ``From`` não estiverem definidos.

    Example:
        >>> class ProductSyncWorkflow(ToThroughFromETLWorkflow):
        ...     From = From([ProductsExtractor])
        ...     Through = Through([CleanProductsTransformer])
        ...     To = To([ProductsLoader])
        ...
        ...     def extract(self, *args, **kwargs):
        ...         extractor = self.From.ProductsExtractor()
        ...         return extractor.extract(
        ...             url_params=URLParams([URLParam("date", self.get_relative_date(-1))])
        ...         )
        ...
        ...     def transform(self, extract_response, *args, **kwargs):
        ...         transformer = self.Through.CleanProductsTransformer()
        ...         return transformer.transform(extract_response)
        ...
        ...     def load(self, transform_response, *args, **kwargs):
        ...         loader = self.To.ProductsLoader()
        ...         return loader.load(data=transform_response)
        ...
        >>> wf = ProductSyncWorkflow()
        >>> wf.run()  # executa extract → transform → load com logs em cada etapa
    """

    To: To = None
    Through: Through = None
    From: From = None

    def __init__(self, *args, **kwargs) -> None:
        assert self.To is not None, \
            "Static attribute To must be set."
        assert self.Through is not None, \
            "Static attribute Through must be set."
        assert self.From is not None, \
            "Static attribute From must be set."
        super(ToThroughFromETLWorkflow, self).__init__(*args, **kwargs)

    def extract(self, *args, **kwargs) -> Any:
        """Etapa de extração. **Deve ser implementado.**

        Returns:
            Dados extraídos para serem passados ao :meth:`transform`.

        Raises:
            NotImplementedError: Sempre que não for sobrescrito.
        """
        raise NotImplementedError

    def transform(self, extract_response: Any, *args, **kwargs) -> Any:
        """Etapa de transformação. **Deve ser implementado.**

        Args:
            extract_response: Resultado retornado por :meth:`extract`.

        Returns:
            Dados transformados para serem passados ao :meth:`load`.

        Raises:
            NotImplementedError: Sempre que não for sobrescrito.
        """
        raise NotImplementedError

    def load(self, transform_response: Any, *args, **kwargs) -> Any:
        """Etapa de carga. **Deve ser implementado.**

        Args:
            transform_response: Resultado retornado por :meth:`transform`.

        Returns:
            Resultado da operação de carga.

        Raises:
            NotImplementedError: Sempre que não for sobrescrito.
        """
        raise NotImplementedError

    def run(self, *args, **kwargs):
        """Orquestra o pipeline ETL completo: extract → transform → load.

        Loga o início e fim de cada etapa. Os resultados intermediários ficam
        disponíveis em ``self.extract_response`` e ``self.transform_response``
        após a execução.

        Args:
            *args: Argumentos repassados a cada etapa.
            **kwargs: Keyword arguments repassados a cada etapa.
        """
        self.LOGGER.info(f"Running workflow: {self.__class__.__name__}...")

        self.LOGGER.info("Extracting...")
        self.extract_response = self.extract(*args, **kwargs)
        self.LOGGER.info("Extracting... Done!")

        self.LOGGER.info("Transforming...")
        self.transform_response = self.transform(self.extract_response, *args, **kwargs)
        self.LOGGER.info("Transforming... Done!")

        self.LOGGER.info("Loading...")
        self.load_response = self.load(self.transform_response, *args, **kwargs)
        self.LOGGER.info("Loading... Done!")

        self.LOGGER.info(f"Running workflow: {self.__class__.__name__}... Done!")


class CustomWorkflow(Workflow):
    """Workflow de estrutura livre para pipelines que não seguem o padrão ETL clássico.

    Em vez das etapas fixas extract/transform/load, o workflow é construído
    dinamicamente pelo método :meth:`construct`, que retorna um callable.

    Subclasses devem implementar :meth:`construct`.

    Example:
        >>> class ReportWorkflow(CustomWorkflow):
        ...     def construct(self, *args, **kwargs):
        ...         def pipeline():
        ...             data = fetch_data()
        ...             report = generate_report(data)
        ...             send_email(report)
        ...             return report
        ...         return pipeline
        ...
        >>> wf = ReportWorkflow()
        >>> pipeline_fn = wf.construct()
        >>> pipeline_fn()
    """

    def construct(self, *args, **kwargs) -> Callable:
        """Constrói e retorna o callable que representa o pipeline customizado.

        **Deve ser implementado pelas subclasses.**

        Returns:
            Callable que encapsula a lógica do workflow.

        Raises:
            NotImplementedError: Sempre que não for sobrescrito.
        """
        raise NotImplementedError
