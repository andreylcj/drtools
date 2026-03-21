

from google.oauth2 import service_account
from googleapiclient.discovery import build
from typing import List, Tuple, Dict, Union
from .types import (
    FileId,
    FilesListResult,
    FilesListItem,
    Mimetype,
)
from .utils import (
    bytes_to_json,
    bytes_to_csv_dicts
)
import io
from googleapiclient.http import (
    MediaIoBaseDownload,
    MediaFileUpload,
    DEFAULT_CHUNK_SIZE,
    MediaIoBaseUpload,
)
from drtools.logging import Logger, FormatterOptions
from typing import Callable
import json
from datetime import datetime


DEFAULTS_FIELDS: str = "nextPageToken, files(id, name, kind, mimeType, createdTime, modifiedTime)"
"""Campos padrão solicitados à API do Drive em todas as listagens.

Inclui token de paginação e os campos de metadados de cada arquivo:
``id``, ``name``, ``kind``, ``mimeType``, ``createdTime``, ``modifiedTime``.
"""

DEFAULT_ORDER_BY: str = "modifiedTime desc"
"""Ordenação padrão das listagens: mais recentemente modificados primeiro."""


class Drive:
    """Cliente de alto nível para a API do Google Drive v3.

    Encapsula autenticação, construção de serviço e as operações mais comuns
    do Drive: listar, baixar, criar e fazer upload de arquivos e pastas.

    A autenticação é delegada ao callable ``credentials_method``, tornando
    a classe flexível para diferentes estratégias (service account, OAuth2, etc.).
    O serviço da API deve ser construído chamando :meth:`build` após a instanciação.

    Class Attributes:
        SCOPES: Escopos OAuth2 usados por padrão (acesso completo ao Drive).

    Args:
        credentials_method: Callable que recebe ``*args, **kwargs`` e retorna
            um objeto de credenciais do Google (e.g.
            ``service_account.Credentials.from_service_account_file``).
        *args: Argumentos posicionais repassados ao ``credentials_method``.
        LOGGER: Logger da drtools. Usa logger padrão ``"Drive"`` se não informado.
        **kwargs: Keyword arguments repassados ao ``credentials_method``.
            O campo ``scopes`` é automaticamente preenchido com :attr:`SCOPES`
            se não fornecido.

    Example:
        >>> from drtools.google.drive import DriveFromServiceAcountFile
        ...
        >>> drive = DriveFromServiceAcountFile("service_account.json")
        >>> drive.build()  # constrói o serviço da API
        ...
        >>> # Listar conteúdo de uma pasta pelo caminho
        >>> result = drive.get_folder_content_from_path("MinhaPasta/Relatorios")
        >>> for f in result["files"]:
        ...     print(f["name"], f["mimeType"])
        ...
        >>> # Baixar um arquivo JSON pelo caminho
        >>> data = drive.get_file_content_from_path("MinhaPasta/dados.json")
        >>> # data já é um dict (parse automático pelo mimeType)
    """

    SCOPES: List[str] = ['https://www.googleapis.com/auth/drive']

    def __init__(
        self,
        credentials_method: Callable,
        *args,
        LOGGER: Logger=None,
        **kwargs,
    ) -> None:
        kwargs['scopes'] = kwargs.get('scopes', self.SCOPES)
        self.credentials = credentials_method(*args, **kwargs)
        if not LOGGER:
            LOGGER = Logger(
                name="Drive",
                formatter_options=FormatterOptions(include_datetime=True, include_logger_name=True, include_level_name=True),
                default_start=False
            )
        self.LOGGER = LOGGER
        self._service = None

    def set_service(self, service) -> None:
        """Define o serviço da API do Drive.

        Normalmente chamado internamente por :meth:`build`.

        Args:
            service: Objeto de serviço retornado por ``googleapiclient.discovery.build``.
        """
        self._service = service

    @property
    def service(self):
        """Serviço da API do Drive. Deve ser inicializado via :meth:`build` antes do uso."""
        return self._service

    def list_files(
        self,
        q: str,
        page_size: int=1000,
        fields: str=DEFAULTS_FIELDS,
        order_by: str=DEFAULT_ORDER_BY,
        page_token: str=None
    ) -> FilesListResult:
        """Executa uma query na API do Drive e retorna arquivos/pastas.

        Wrapper direto sobre ``service.files().list()``.

        Args:
            q: Query no formato da Drive API
                (ex: ``"'folderId' in parents and trashed=false"``).
            page_size: Número máximo de itens por página. Padrão: ``1000``.
            fields: Campos a retornar. Padrão: :data:`DEFAULTS_FIELDS`.
            order_by: Campo e direção de ordenação. Padrão: :data:`DEFAULT_ORDER_BY`.
            page_token: Token para buscar página específica. ``None`` para primeira página.

        Returns:
            :class:`~drtools.google.drive.types.FilesListResult` com os arquivos
            encontrados e o token da próxima página (se houver).

        Example:
            >>> result = drive.list_files(
            ...     q="name='relatorio.csv' and trashed=false"
            ... )
            >>> result["files"]
        """
        return self.service.files().list(
            q=q,
            pageSize=page_size,
            fields=fields,
            orderBy=order_by,
            pageToken=page_token,
        ).execute()

    def get_folders_from_name(
        self,
        name: str,
        page_size: int=10,
        fields: str=DEFAULTS_FIELDS,
        parent_folder_id: str=None,
        order_by: str=DEFAULT_ORDER_BY,
        page_token: str=None
    ) -> FilesListResult:
        """Busca pastas pelo nome, opcionalmente restringindo a uma pasta pai.

        Args:
            name: Nome exato da pasta a buscar.
            page_size: Número máximo de resultados. Padrão: ``10``.
            fields: Campos a retornar. Padrão: :data:`DEFAULTS_FIELDS`.
            parent_folder_id: ID da pasta pai. Se informado, restringe a busca
                somente a filhos diretos dessa pasta.
            order_by: Ordenação. Padrão: :data:`DEFAULT_ORDER_BY`.
            page_token: Token de paginação.

        Returns:
            :class:`~drtools.google.drive.types.FilesListResult` com as pastas encontradas.

        Example:
            >>> # Buscar pasta "2026" dentro de uma pasta pai
            >>> result = drive.get_folders_from_name("2026", parent_folder_id="abc123")
            >>> folder_id = result["files"][0]["id"]
        """
        q = f"mimeType='application/vnd.google-apps.folder' and name='{name}'"
        if parent_folder_id:
            q += f" and '{parent_folder_id}' in parents"
        results = self.list_files(
            q=q,
            page_size=page_size,
            fields=fields,
            order_by=order_by,
            page_token=page_token,
        )
        return results

    def get_folder_content(
        self,
        folder_id,
        page_size: int=1000,
        trashed: bool=False,
        fields: str=DEFAULTS_FIELDS,
        deep: bool=False,
        order_by: str=DEFAULT_ORDER_BY,
        page_token: str=None,
        custom_query_extend_and_case: str=None
    ) -> FilesListResult:
        """Lista o conteúdo de uma pasta pelo seu ID.

        Args:
            folder_id: ID da pasta no Google Drive.
            page_size: Número máximo de itens por página. Padrão: ``1000``.
            trashed: Se ``True``, inclui itens na lixeira. Padrão: ``False``.
            fields: Campos a retornar. Padrão: :data:`DEFAULTS_FIELDS`.
            deep: Se ``True``, percorre recursivamente as subpastas, adicionando
                um campo ``"content"`` a cada item do tipo pasta. Padrão: ``False``.
            order_by: Ordenação. Padrão: :data:`DEFAULT_ORDER_BY`.
            page_token: Token de paginação.
            custom_query_extend_and_case: Trecho de query adicional concatenado
                com ``" and "`` após a query padrão. Útil para filtrar por
                ``mimeType``, ``modifiedTime``, etc.

        Returns:
            :class:`~drtools.google.drive.types.FilesListResult`. Quando ``deep=True``,
            cada item do tipo pasta terá um campo ``"content"`` com o resultado
            recursivo.

        Example:
            >>> # Listar arquivos CSV de uma pasta
            >>> result = drive.get_folder_content(
            ...     folder_id="abc123",
            ...     custom_query_extend_and_case="mimeType='text/csv'"
            ... )
            >>> for f in result["files"]:
            ...     print(f["name"])

            >>> # Listar recursivamente
            >>> result = drive.get_folder_content(folder_id="abc123", deep=True)
        """
        def _get_folder_content(folder_id, page_size, trashed):
            trashed = 'true' if trashed else 'false'
            q = f"'{folder_id}' in parents and trashed={trashed}"
            if custom_query_extend_and_case:
                q += ' and ' + custom_query_extend_and_case
            results = self.list_files(
                q=q,
                page_size=page_size,
                fields=fields,
                order_by=order_by,
                page_token=page_token,
            )
            items = results
            return items
        items = _get_folder_content(folder_id, page_size, trashed)
        if deep:
            items = [
                {
                    **item,
                    'content': self.get_folder_content(
                            folder_id=item['id'],
                            page_size=page_size,
                            trashed=trashed,
                            fields=fields,
                            deep=deep,
                            order_by=order_by,
                            page_token=page_token,
                        ) if 'folder' in item['mimeType'] else None
                } for item in items
            ]
        return items

    def list_files_from_folder_filtering_by_modified_time(
        self,
        path: str,
        page_size: int=1000,
        trashed: bool=False,
        fields: str=DEFAULTS_FIELDS,
        deep: bool=False,
        order_by: str=DEFAULT_ORDER_BY,
        page_token: str=None,
        start_date: str=None,
        end_date: str=None,
    ) -> FilesListResult:
        """Lista arquivos de uma pasta filtrados por data de modificação.

        Converte as datas ISO para o formato RFC 3339 exigido pela Drive API
        e delega para :meth:`get_folder_content_from_path` com a query de filtro
        correspondente.

        Args:
            path: Caminho da pasta no Drive (e.g. ``"Projetos/2026/Janeiro"``).
            page_size: Número máximo de itens por página. Padrão: ``1000``.
            trashed: Se ``True``, inclui itens na lixeira. Padrão: ``False``.
            fields: Campos a retornar. Padrão: :data:`DEFAULTS_FIELDS`.
            deep: Se ``True``, percorre subpastas recursivamente. Padrão: ``False``.
            order_by: Ordenação. Padrão: :data:`DEFAULT_ORDER_BY`.
            page_token: Token de paginação.
            start_date: Data de início no formato ISO (``"YYYY-MM-DD"`` ou
                ``"YYYY-MM-DDTHH:MM:SS"``). Filtra ``modifiedTime >= start_date``.
            end_date: Data de fim no formato ISO. Filtra ``modifiedTime <= end_date``.

        Returns:
            :class:`~drtools.google.drive.types.FilesListResult` com os arquivos
            modificados no intervalo informado.

        Example:
            >>> # Arquivos modificados entre 1 e 20 de março de 2026
            >>> result = drive.list_files_from_folder_filtering_by_modified_time(
            ...     path="Projetos/Relatorios",
            ...     start_date="2026-03-01",
            ...     end_date="2026-03-20",
            ... )
            >>> for f in result["files"]:
            ...     print(f["name"], f["modifiedTime"])

            >>> # Apenas arquivos modificados a partir de uma data
            >>> result = drive.list_files_from_folder_filtering_by_modified_time(
            ...     path="Projetos/Relatorios",
            ...     start_date="2026-03-15",
            ... )
        """
        start_date_dt = None
        end_date_dt = None

        if start_date:
            start_date_dt = datetime.fromisoformat(start_date)

        if end_date:
            end_date_dt = datetime.fromisoformat(end_date)

        custom_query_extend_and_case = ''

        if start_date_dt and end_date_dt:
            start_rfc3339  = start_date_dt.isoformat() + 'Z'
            end_rfc3339 = end_date_dt.isoformat() + 'Z'
            custom_query_extend_and_case += f"modifiedTime >= '{start_rfc3339}' and modifiedTime <= '{end_rfc3339}'"

        elif start_date_dt and not end_date_dt:
            start_rfc3339  = start_date_dt.isoformat() + 'Z'
            custom_query_extend_and_case += f"modifiedTime >= '{start_rfc3339}'"

        elif not start_date_dt and end_date_dt:
            end_rfc3339  = end_date_dt.isoformat() + 'Z'
            custom_query_extend_and_case += f"modifiedTime <= '{end_rfc3339}'"

        return self.get_folder_content_from_path(
            path=path,
            page_size=page_size,
            trashed=trashed,
            fields=fields,
            deep=deep,
            order_by=order_by,
            page_token=page_token,
            custom_query_extend_and_case=custom_query_extend_and_case,
        )

    def get_folder_id_from_path(self, path: str) -> FileId:
        """Resolve um caminho de pasta (separado por ``/``) para o seu ID no Drive.

        Percorre cada segmento do caminho sequencialmente, verificando unicidade
        em cada nível. O segmento raiz deve ser único em todo o Drive; os demais
        devem ser únicos dentro do pai.

        Args:
            path: Caminho da pasta usando ``/`` como separador
                (e.g. ``"Projetos/2026/Janeiro"``).

        Returns:
            :class:`~drtools.google.drive.types.FileId` da pasta final do caminho.

        Raises:
            Exception: Se a raiz do caminho não for única no Drive.
            Exception: Se um segmento não for único dentro do seu pai.
            Exception: Se alguma pasta do caminho não for encontrada.

        Example:
            >>> folder_id = drive.get_folder_id_from_path("Projetos/2026/Janeiro")
            >>> print(folder_id)  # "1BxiMVs0XRA5..."
        """
        folder_names = path.split('/')
        parent_id = None
        parent_path = None
        for idx, folder_name in enumerate(folder_names):
            if idx == 0:
                results = self.get_folders_from_name(folder_name)
            else:
                results = self.get_folders_from_name(folder_name, parent_folder_id=parent_id)
            results_len = len(results['files'])
            if results_len > 1:
                if idx == 0:
                    raise Exception(f"Path root must be unique in Drive. Files results for {folder_name} were {results_len:,}.")
                else:
                    raise Exception(f"Folder name must be unique inside parent. Folder name: {folder_name} | Parent path: {parent_path} | Parent ID: {parent_id}")
            if results_len == 0:
                raise Exception(f"Folder not find. Folder name: {folder_name} | Parent path: {parent_path} | Parent ID: {parent_id}")
            parent_id = results['files'][0]['id']
            parent_path = folder_name if idx == 0 else f'{parent_path}/{folder_name}'
        return parent_id

    def get_folder_content_from_path(
        self,
        path: str,
        page_size: int=1000,
        trashed: bool=False,
        fields: str=DEFAULTS_FIELDS,
        deep: bool=False,
        order_by: str=DEFAULT_ORDER_BY,
        page_token: str=None,
        custom_query_extend_and_case: str=None
    ) -> FilesListResult:
        """Lista o conteúdo de uma pasta a partir do seu caminho no Drive.

        Resolve o caminho para um ID via :meth:`get_folder_id_from_path` e
        delega para :meth:`get_folder_content`.

        Args:
            path: Caminho da pasta (e.g. ``"Projetos/2026"``).
            page_size: Número máximo de itens por página. Padrão: ``1000``.
            trashed: Se ``True``, inclui itens na lixeira. Padrão: ``False``.
            fields: Campos a retornar. Padrão: :data:`DEFAULTS_FIELDS`.
            deep: Se ``True``, percorre subpastas recursivamente. Padrão: ``False``.
            order_by: Ordenação. Padrão: :data:`DEFAULT_ORDER_BY`.
            page_token: Token de paginação.
            custom_query_extend_and_case: Trecho de query adicional concatenado
                com ``" and "`` após a query padrão.

        Returns:
            :class:`~drtools.google.drive.types.FilesListResult`.

        Example:
            >>> result = drive.get_folder_content_from_path("Projetos/2026")
            >>> for f in result["files"]:
            ...     print(f["name"], f["id"])
        """
        folder_id = self.get_folder_id_from_path(path)
        return self.get_folder_content(
            folder_id=folder_id,
            page_size=page_size,
            trashed=trashed,
            fields=fields,
            deep=deep,
            order_by=order_by,
            page_token=page_token,
            custom_query_extend_and_case=custom_query_extend_and_case,
        )

    def create_folder(
        self,
        folder_name: str,
        parent_folder_id: FileId,
        ignore_if_exists: bool=True
    ) -> FileId:
        """Cria uma pasta no Google Drive dentro de um pai especificado por ID.

        Args:
            folder_name: Nome da nova pasta.
            parent_folder_id: ID da pasta pai onde a nova pasta será criada.
            ignore_if_exists: Se ``True`` (padrão), retorna ``None`` silenciosamente
                se já existir uma pasta/arquivo com o mesmo nome no pai.
                Se ``False``, cria duplicata.

        Returns:
            ID da pasta criada, ou ``None`` se já existia e ``ignore_if_exists=True``.

        Example:
            >>> parent_id = drive.get_folder_id_from_path("Projetos")
            >>> new_id = drive.create_folder("2026", parent_id)
            >>> print(new_id)  # ID da nova pasta, ou None se já existia
        """
        folder_metadata = {
            'name': folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            'parents': [parent_folder_id] if parent_folder_id else []
        }
        if ignore_if_exists:
            if self.children_exists(folder_name, parent_folder_id):
                return None
        created_folder = self.service.files().create(body=folder_metadata, fields='id').execute()
        return created_folder["id"]

    def create_folder_from_path(
        self,
        folder_path: str,
        ignore_if_exists: bool=True
    ) -> FileId:
        """Cria uma pasta usando um caminho completo no Drive.

        O caminho deve ter pelo menos um nível pai já existente. Apenas o
        último segmento do caminho é criado; os anteriores são resolvidos.

        Args:
            folder_path: Caminho completo incluindo a nova pasta
                (e.g. ``"Projetos/2026/Março"`` — cria ``"Março"`` dentro de
                ``"Projetos/2026"``).
            ignore_if_exists: Se ``True`` (padrão), não cria se já existe.

        Returns:
            ID da pasta criada, ou ``None`` se já existia e ``ignore_if_exists=True``.

        Example:
            >>> new_id = drive.create_folder_from_path("Projetos/2026/Marco")
        """
        parent, name = self.get_parent_and_name_from_path(folder_path)
        parent_id = self.get_folder_id_from_path(parent)
        return self.create_folder(name, parent_id, ignore_if_exists)

    def get_file_from_path(self, path: str) -> FilesListItem:
        """Retorna os metadados de um arquivo a partir do seu caminho no Drive.

        Resolve a pasta pai e busca o arquivo pelo nome dentro dela.

        Args:
            path: Caminho completo do arquivo (e.g. ``"Projetos/dados.csv"``).

        Returns:
            :class:`~drtools.google.drive.types.FilesListItem` com os metadados do arquivo.

        Raises:
            Exception: Se mais de um arquivo com o mesmo nome for encontrado.
            Exception: Se nenhum arquivo for encontrado.

        Example:
            >>> item = drive.get_file_from_path("Projetos/dados.csv")
            >>> print(item["id"], item["mimeType"])
        """
        folder_path, filename = self.get_parent_and_name_from_path(path)
        folder_content = self.get_folder_content_from_path(folder_path)
        file_item = [item for item in folder_content['files'] if item['name'] == filename]
        if len(file_item) > 1:
            raise Exception(f"There are more than 1 file with same path. Files founde: {len(file_item):,}")
        if len(file_item) == 0:
            raise Exception(f"No file found with path: {path}")
        return file_item[0]

    def get_file_id_from_path(self, path: str) -> FileId:
        """Retorna o ID de um arquivo a partir do seu caminho no Drive.

        Atalho para ``get_file_from_path(path)["id"]``.

        Args:
            path: Caminho completo do arquivo (e.g. ``"Projetos/dados.csv"``).

        Returns:
            :class:`~drtools.google.drive.types.FileId` do arquivo.

        Example:
            >>> file_id = drive.get_file_id_from_path("Projetos/dados.csv")
        """
        return self.get_file_from_path(path)['id']

    def get_file_content(
        self,
        file_id: str,
        try_handle_mimetype: bool=True,
        mimetype: str=None,
        **handle_bytes_from_mimetype_kwargs
    ) -> bytes:
        """Baixa o conteúdo de um arquivo pelo ID, opcionalmente fazendo parse pelo MIME type.

        Utiliza ``MediaIoBaseDownload`` para download em chunks e, se
        ``try_handle_mimetype=True`` e ``mimetype`` for fornecido, chama
        :meth:`handle_bytes_from_mimetype` para converter os bytes no tipo
        Python correspondente (dict para JSON, list of dicts para CSV).

        Args:
            file_id: ID do arquivo no Drive.
            try_handle_mimetype: Se ``True`` e ``mimetype`` fornecido, faz parse
                automático do conteúdo. Padrão: ``True``.
            mimetype: Tipo MIME do arquivo. Se ``None``, retorna bytes brutos.
            **handle_bytes_from_mimetype_kwargs: Kwargs adicionais repassados
                a :meth:`handle_bytes_from_mimetype` (e.g. ``encoding``,
                ``delimiter`` para CSV).

        Returns:
            Bytes brutos do arquivo, ou o objeto Python resultante do parse
            (dict, list, etc.) se ``try_handle_mimetype=True``.

        Example:
            >>> # Baixar como bytes brutos
            >>> content = drive.get_file_content("abc123", try_handle_mimetype=False)
            ...
            >>> # Baixar e parsear CSV automaticamente
            >>> records = drive.get_file_content(
            ...     "abc123",
            ...     mimetype="text/csv",
            ...     delimiter=";",
            ... )
            >>> isinstance(records, list)
            True
        """
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            self.LOGGER.debug(f"[GoogleDrive:FileID:{file_id}] Download {int(status.progress() * 100)}%.")
        value = fh.getvalue()
        if try_handle_mimetype and mimetype:
            value = self.handle_bytes_from_mimetype(value, mimetype, **handle_bytes_from_mimetype_kwargs)
        return value

    @classmethod
    def get_parent_and_name_from_path(cls, filepath: str) -> Tuple[str, str]:
        """Divide um caminho no Drive em pasta pai e nome do item final.

        Args:
            filepath: Caminho completo (e.g. ``"Projetos/2026/dados.csv"``).

        Returns:
            Tupla ``(parent, name)`` onde ``parent`` é o caminho da pasta pai
            e ``name`` é o nome do item (último segmento).

        Example:
            >>> Drive.get_parent_and_name_from_path("Projetos/2026/dados.csv")
            ('Projetos/2026', 'dados.csv')
            >>> Drive.get_parent_and_name_from_path("Projetos/2026")
            ('Projetos', '2026')
        """
        parent = '/'.join(filepath.split('/')[:-1])
        name = filepath.split('/')[-1]
        return parent, name

    def children_exists(self, name: str, parent_folder_id: str) -> bool:
        """Verifica se existe um item com o nome informado dentro de uma pasta.

        Args:
            name: Nome do arquivo ou pasta a verificar.
            parent_folder_id: ID da pasta pai.

        Returns:
            ``True`` se encontrado, ``False`` caso contrário.

        Example:
            >>> exists = drive.children_exists("relatorio.csv", "abc123")
            >>> if not exists:
            ...     drive.create_file(...)
        """
        content = self.get_folder_content(parent_folder_id)
        file_item = [item for item in content['files'] if item['name'] == name]
        return len(file_item) > 0

    def path_exists(self, path: str) -> bool:
        """Verifica se um caminho (arquivo ou pasta) existe no Drive.

        Resolve a pasta pai pelo caminho e busca o item pelo nome dentro dela.

        Args:
            path: Caminho completo a verificar (e.g. ``"Projetos/2026/dados.csv"``).

        Returns:
            ``True`` se o item existir, ``False`` caso contrário.

        Example:
            >>> if not drive.path_exists("Projetos/2026/dados.csv"):
            ...     drive.create_file_from_media_io(content, "Projetos/2026/dados.csv", "text/csv")
        """
        parent, name = self.get_parent_and_name_from_path(path)
        content = self.get_folder_content_from_path(parent)
        file_item = [item for item in content['files'] if item['name'] == name]
        return len(file_item) > 0

    def get_file_content_from_path(
        self,
        filepath: str,
        try_handle_mimetype: bool=True,
    ) -> io.BytesIO:
        """Baixa o conteúdo de um arquivo a partir do seu caminho no Drive.

        Resolve o caminho para o ID do arquivo e delega para :meth:`get_file_content`,
        passando automaticamente o ``mimeType`` retornado pela API.

        Args:
            filepath: Caminho completo do arquivo (e.g. ``"Projetos/dados.json"``).
            try_handle_mimetype: Se ``True`` (padrão), faz parse automático do
                conteúdo conforme o ``mimeType`` do arquivo.

        Returns:
            Bytes brutos, dict (JSON), ou list of dicts (CSV) dependendo do
            ``mimeType`` e do valor de ``try_handle_mimetype``.

        Raises:
            Exception: Se mais de um arquivo com o mesmo nome for encontrado.
            Exception: Se nenhum arquivo for encontrado.

        Example:
            >>> # Arquivo JSON: retorna dict automaticamente
            >>> config = drive.get_file_content_from_path("Configs/settings.json")
            >>> config["timeout"]
            30

            >>> # Arquivo CSV: retorna list of dicts automaticamente
            >>> rows = drive.get_file_content_from_path("Dados/clientes.csv")
            >>> rows[0]["nome"]
            'Alice'

            >>> # Forçar retorno como bytes brutos
            >>> raw = drive.get_file_content_from_path("Dados/imagem.png", try_handle_mimetype=False)
        """
        folder_path, filename = self.get_parent_and_name_from_path(filepath)
        self_folder_content = self.get_folder_content_from_path(folder_path)
        file_item = [item for item in self_folder_content['files'] if item['name'] == filename]
        if len(file_item) > 1:
            raise Exception(f"There are more than 1 file with same path. Files founde: {len(file_item):,}")
        if len(file_item) == 0:
            raise Exception(f"No file found with path: {filepath}")
        file_id = file_item[0]['id']
        return self.get_file_content(file_id, try_handle_mimetype, file_item[0]['mimeType'])

    def get_last_modified_file_content_from_folder(
        self,
        folder_path: str,
        mimetype: str=None
    ):
        """Baixa o conteúdo do arquivo modificado mais recentemente em uma pasta.

        Usa ``page_size=1`` com ordenação por ``modifiedTime desc`` para buscar
        apenas o arquivo mais recente.

        Args:
            folder_path: Caminho da pasta no Drive (e.g. ``"Exports/Daily"``).
            mimetype: Tipo MIME para parse. Se ``None``, usa o ``mimeType``
                retornado pela API.

        Returns:
            Conteúdo do arquivo (bytes, dict ou list of dicts conforme o MIME type).

        Example:
            >>> # Pegar o CSV mais recente de uma pasta de exports
            >>> latest = drive.get_last_modified_file_content_from_folder(
            ...     "Exports/Daily"
            ... )
            >>> latest[0]["date"]
            '2026-03-20'
        """
        folder_content = self.get_folder_content_from_path(folder_path, page_size=1)
        file_id = folder_content['files'][0]['id']
        mimetype = folder_content['files'][0]['mimeType']
        data = self.get_file_content(file_id, mimetype=mimetype)
        return data

    def create_file(
        self,
        name: str,
        parent_folder_id: FileId,
        media: Union[MediaFileUpload, MediaIoBaseUpload]=None,
        ignore_if_exists: bool=True
    ) -> FileId:
        """Cria um arquivo no Google Drive dentro de uma pasta especificada por ID.

        Args:
            name: Nome do arquivo a criar.
            parent_folder_id: ID da pasta pai.
            media: Objeto de mídia para upload
                (:class:`googleapiclient.http.MediaFileUpload` ou
                :class:`googleapiclient.http.MediaIoBaseUpload`).
                Se ``None``, cria um arquivo vazio.
            ignore_if_exists: Se ``True`` (padrão), não cria e loga um aviso
                se já existir um item com o mesmo nome na pasta pai.

        Returns:
            ID do arquivo criado, ou ``None`` se já existia e ``ignore_if_exists=True``.

        Example:
            >>> import io
            >>> from googleapiclient.http import MediaIoBaseUpload
            >>> content = b"hello world"
            >>> stream = io.BytesIO(content)
            >>> media = MediaIoBaseUpload(stream, mimetype="text/plain")
            >>> parent_id = drive.get_folder_id_from_path("Projetos")
            >>> file_id = drive.create_file("hello.txt", parent_id, media)
        """
        file_metadata = {
            'name': name,
            'parents': [parent_folder_id] if parent_folder_id else []
        }
        if ignore_if_exists:
            if self.children_exists(name, parent_folder_id):
                self.LOGGER.debug(f'File with name {name} inside folder with id {parent_folder_id} already exists')
                return None
        kwargs = {'body': file_metadata, 'fields': 'id'}
        if media:
            kwargs['media_body'] = media
        created_file = self.service.files().create(**kwargs).execute()
        return created_file["id"]

    def create_file_from_media_file(
        self,
        filepath: str,
        filename: str,
        mimetype: str=None,
        chunksize=DEFAULT_CHUNK_SIZE,
        resumable=False,
        ignore_if_exists: bool=True
    ) -> FileId:
        """Faz upload de um arquivo local para o Drive usando um caminho de destino.

        Resolve o caminho de destino para o ID da pasta pai e usa
        :class:`googleapiclient.http.MediaFileUpload` para o upload.

        Args:
            filepath: Caminho de destino no Drive
                (e.g. ``"Projetos/dados.csv"`` — a pasta ``"Projetos"`` deve existir).
            filename: Caminho do arquivo local a ser enviado.
            mimetype: Tipo MIME do arquivo. Se ``None``, o cliente detecta automaticamente.
            chunksize: Tamanho de cada chunk de upload. Padrão: ``DEFAULT_CHUNK_SIZE``.
            resumable: Se ``True``, usa upload resumível. Padrão: ``False``.
            ignore_if_exists: Se ``True`` (padrão), não envia se já existe arquivo
                com o mesmo nome no destino.

        Returns:
            ID do arquivo criado, ou ``None`` se já existia e ``ignore_if_exists=True``.

        Example:
            >>> file_id = drive.create_file_from_media_file(
            ...     filepath="Projetos/relatorio.csv",
            ...     filename="/tmp/relatorio.csv",
            ...     mimetype="text/csv",
            ... )
        """
        parent, name = self.get_parent_and_name_from_path(filepath)
        parent_id = self.get_folder_id_from_path(parent)
        media = MediaFileUpload(filename, mimetype, chunksize, resumable)
        return self.create_file(name, parent_id, media, ignore_if_exists)

    def create_file_from_media_io(
        self,
        content_bytes: bytes,
        filepath: str,
        mimetype: str,
        chunksize=DEFAULT_CHUNK_SIZE,
        resumable=False,
        ignore_if_exists: bool=True
    ) -> FileId:
        """Faz upload de conteúdo em bytes para o Drive usando um caminho de destino.

        Encapsula os bytes em :class:`io.BytesIO` e usa
        :class:`googleapiclient.http.MediaIoBaseUpload` para o upload.

        Args:
            content_bytes: Conteúdo do arquivo como bytes.
            filepath: Caminho de destino no Drive (e.g. ``"Exports/output.json"``).
            mimetype: Tipo MIME do conteúdo (e.g. ``"application/json"``, ``"text/csv"``).
            chunksize: Tamanho de cada chunk. Padrão: ``DEFAULT_CHUNK_SIZE``.
            resumable: Se ``True``, usa upload resumível. Padrão: ``False``.
            ignore_if_exists: Se ``True`` (padrão), não envia se já existe.

        Returns:
            ID do arquivo criado, ou ``None`` se já existia e ``ignore_if_exists=True``.

        Example:
            >>> csv_content = "nome,idade\\nAlice,30".encode("utf-8")
            >>> file_id = drive.create_file_from_media_io(
            ...     content_bytes=csv_content,
            ...     filepath="Exports/clientes.csv",
            ...     mimetype="text/csv",
            ... )
        """
        parent, name = self.get_parent_and_name_from_path(filepath)
        parent_id = self.get_folder_id_from_path(parent)
        content_stream = io.BytesIO(content_bytes)
        media = MediaIoBaseUpload(content_stream, mimetype, chunksize, resumable)
        return self.create_file(name, parent_id, media, ignore_if_exists)

    def upload_dict(
        self,
        data: Dict,
        filepath: str,
        chunksize=DEFAULT_CHUNK_SIZE,
        resumable=False,
        ignore_if_exists: bool=True,
    ):
        """Serializa um dicionário como JSON e faz upload para o Drive.

        Atalho para ``create_file_from_media_io`` com MIME type ``application/json``.

        Args:
            data: Dicionário Python a ser serializado e enviado.
            filepath: Caminho de destino no Drive (e.g. ``"Configs/settings.json"``).
            chunksize: Tamanho de cada chunk. Padrão: ``DEFAULT_CHUNK_SIZE``.
            resumable: Se ``True``, usa upload resumível. Padrão: ``False``.
            ignore_if_exists: Se ``True`` (padrão), não envia se já existe.

        Returns:
            ID do arquivo criado, ou ``None`` se já existia e ``ignore_if_exists=True``.

        Example:
            >>> config = {"timeout": 30, "retries": 3, "env": "prod"}
            >>> file_id = drive.upload_dict(
            ...     data=config,
            ...     filepath="Configs/settings.json",
            ... )
        """
        content_bytes = json.dumps(data).encode('utf-8')
        return self.create_file_from_media_io(
            content_bytes,
            filepath,
            Mimetype.JSON.content_type,
            chunksize,
            resumable,
            ignore_if_exists
        )

    @classmethod
    def handle_bytes_from_mimetype(
        cls,
        content: bytes,
        mimetype: str,
        raise_exception: bool=True,
        **kwargs
    ):
        """Converte bytes para o tipo Python correspondente ao MIME type informado.

        Suporta os tipos definidos em :class:`~drtools.google.drive.types.Mimetype`:

        - ``application/json`` → dict via :func:`~drtools.google.drive.utils.bytes_to_json`
        - ``text/csv`` → list of dicts via :func:`~drtools.google.drive.utils.bytes_to_csv_dicts`

        Args:
            content: Conteúdo binário do arquivo.
            mimetype: Tipo MIME que determina como os bytes serão interpretados.
            raise_exception: Se ``True`` (padrão), lança exceção para MIME types
                não suportados. Se ``False``, retorna os bytes brutos.
            **kwargs: Argumentos adicionais repassados às funções de conversão
                (e.g. ``encoding``, ``delimiter`` para CSV).

        Returns:
            - ``dict`` para JSON
            - ``list[dict]`` para CSV
            - ``bytes`` se ``raise_exception=False`` e MIME não suportado

        Raises:
            Exception: Se ``mimetype`` não for suportado e ``raise_exception=True``.

        Example:
            >>> import json
            >>> json_bytes = json.dumps({"a": 1}).encode("utf-8")
            >>> Drive.handle_bytes_from_mimetype(json_bytes, "application/json")
            {'a': 1}

            >>> csv_bytes = b"nome,idade\\nAlice,30"
            >>> Drive.handle_bytes_from_mimetype(csv_bytes, "text/csv")
            [{'nome': 'Alice', 'idade': '30'}]

            >>> Drive.handle_bytes_from_mimetype(b"raw", "image/png", raise_exception=False)
            b'raw'
        """
        if mimetype == Mimetype.JSON.content_type:
            value = bytes_to_json(content, **kwargs)

        elif mimetype == Mimetype.CSV.content_type:
            value = bytes_to_csv_dicts(content, **kwargs)

        else:
            if raise_exception:
                raise Exception(f"Mime Type {mimetype} not allow yet.")
            value = content

        return value

    def build(self, *args, **kwargs):
        """Constrói o serviço da API do Drive. **Deve ser implementado pelas subclasses.**

        Raises:
            NotImplementedError: Sempre que não for sobrescrito.
        """
        raise NotImplementedError


class DriveFromServiceAcountFile(Drive):
    """Implementação de :class:`Drive` autenticada via arquivo de service account JSON.

    Usa ``google.oauth2.service_account.Credentials.from_service_account_file``
    para carregar as credenciais do arquivo informado.

    Args:
        filename: Caminho para o arquivo JSON de service account.
        **kwargs: Keyword arguments adicionais repassados à classe pai
            (e.g. ``scopes``, ``LOGGER``).

    Example:
        >>> from drtools.google.drive import DriveFromServiceAcountFile
        ...
        >>> drive = DriveFromServiceAcountFile("service_account.json")
        >>> drive.build()  # versão padrão v3
        ...
        >>> # Verificar existência de um arquivo
        >>> if drive.path_exists("Projetos/dados.csv"):
        ...     data = drive.get_file_content_from_path("Projetos/dados.csv")
        ...
        >>> # Fazer upload de um dict como JSON
        >>> drive.upload_dict(
        ...     data={"status": "ok", "records": 150},
        ...     filepath="Logs/resultado.json",
        ... )
        ...
        >>> # Listar arquivos modificados nos últimos 7 dias
        >>> from datetime import datetime, timedelta
        >>> start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        >>> result = drive.list_files_from_folder_filtering_by_modified_time(
        ...     path="Exports/Daily",
        ...     start_date=start,
        ... )
        >>> for f in result["files"]:
        ...     print(f["name"], f["modifiedTime"])
    """

    def __init__(self, filename: str, **kwargs) -> None:
        super(DriveFromServiceAcountFile, self).__init__(
            service_account.Credentials.from_service_account_file,
            filename,
            **kwargs
        )

    def build(
        self,
        version: str='v3',
        *args,
        **kwargs
    ):
        """Constrói o serviço da API do Google Drive.

        Args:
            version: Versão da API do Drive. Padrão: ``"v3"``.
            *args: Argumentos posicionais adicionais repassados ao
                ``googleapiclient.discovery.build``.
            **kwargs: Keyword arguments adicionais repassados ao
                ``googleapiclient.discovery.build``.

        Example:
            >>> drive = DriveFromServiceAcountFile("service_account.json")
            >>> drive.build()          # usa v3 por padrão
            >>> drive.build("v2")     # usa v2
        """
        self.LOGGER.info("Building service...")
        kwargs['credentials'] = self.credentials
        self.set_service(build('drive', version, *args, **kwargs))
        self.LOGGER.info("Building service... Done!")
