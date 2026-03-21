

from .exceptions import BotDetectionError
from typing import List


class BotDetection:
    """Classe base para detectores de bot/bloqueio em páginas web.

    Define o contrato para verificação de páginas via XPath. Subclasses devem
    definir :attr:`CONTENT` com os textos que indicam detecção de bot.

    O método :meth:`detect` é chamado por
    :meth:`~drtools.web_automation.driver_handler.handler.WebDriverHandler.go_to_page`
    após cada carregamento de página.

    Class Attributes:
        CONTENT: Lista de strings cujo aparecimento na página indica detecção de bot.
            **Obrigatório nas subclasses.**
        MESSAGE: Mensagem da exceção levantada ao detectar bot.

    Example:
        >>> class MyBotDetection(BotDetection):
        ...     CONTENT = ["Acesso bloqueado", "Você foi detectado como bot"]
        ...     MESSAGE = "Detecção de bot customizada."
        ...
        >>> # Usar no handler:
        >>> handler = ChromeWebDriverHandler(
        ...     bot_detection_methods=[MyBotDetection]
        ... )
        >>> handler.start(remove_ui=True)
        >>> handler.go_to_page("https://example.com")  # detecta e retry automaticamente
    """

    CONTENT: List[str] = None
    MESSAGE: str = 'Bot detection.'

    @classmethod
    def build_pattern(cls) -> str:
        """Constrói o XPath que busca qualquer um dos textos de :attr:`CONTENT` na página.

        Returns:
            String XPath no formato ``//*[contains(., "texto1") or contains(., "texto2")]``.

        Example:
            >>> class MyDetection(BotDetection):
            ...     CONTENT = ["blocked", "captcha"]
            >>> MyDetection.build_pattern()
            '//*[contains(., "blocked") or contains(., "captcha")]'
        """
        return f'//*[' + ' or '.join([f'contains(., "{s}")' for s in cls.CONTENT]) + ']'

    @classmethod
    def detect(
        cls,
        web_drive_handler,
        raise_exception: bool=True
    ) -> None:
        """Executa a detecção de bot na página atual do driver.

        Busca pelo XPath construído via :meth:`build_pattern`. Se encontrar
        o elemento, loga o erro e levanta :class:`~drtools.web_automation.exceptions.BotDetectionError`.

        Args:
            web_drive_handler: Instância de
                :class:`~drtools.web_automation.driver_handler.handler.WebDriverHandler`
                com o driver ativo.
            raise_exception: Se ``True`` (padrão), propaga a exceção após logar.
                Se ``False``, apenas loga.

        Raises:
            BotDetectionError: Se os textos de :attr:`CONTENT` forem encontrados na página.
        """
        pattern = cls.build_pattern()
        try:
            bot_el = web_drive_handler.wait_for_element_presence_located_by_xpath(
                pattern,
                web_drive_handler.bot_detection_wait_for_presence_delay,
                False
            )
            if bot_el:
                raise BotDetectionError(cls.MESSAGE)
        except BotDetectionError as exc:
            web_drive_handler.LOGGER.error(str(exc))
            if raise_exception:
                raise exc


class GoogleBotDetection(BotDetection):
    """Detecta a tela de verificação de tráfego incomum do Google.

    Ativada quando o Google exibe mensagem de tráfego anômalo
    (tipicamente antes de um CAPTCHA).

    Example:
        >>> handler = ChromeWebDriverHandler(
        ...     bot_detection_methods=[GoogleBotDetection]
        ... )
    """

    CONTENT: List[str] = [
        "Nossos sistemas detectaram tráfego incomum na sua rede de computadores",
    ]
    MESSAGE: str = 'Google bot detection.'


class BlockDetection(BotDetection):
    """Detecta bloqueios do tipo Cloudflare "Sorry, you have been blocked".

    Example:
        >>> handler = ChromeWebDriverHandler(
        ...     bot_detection_methods=[BlockDetection]
        ... )
    """

    CONTENT: List[str] = [
        "Sorry, you have been blocked",
    ]
    MESSAGE: str = 'Block detected.'


class HumanDetection(BotDetection):
    """Detecta telas de verificação de humanidade (CAPTCHA em português).

    Ativada quando a página exibe verificações como "Confirme que você é humano"
    ou "Verificando se você é humano".

    Example:
        >>> handler = ChromeWebDriverHandler(
        ...     bot_detection_methods=[HumanDetection]
        ... )
    """

    CONTENT: List[str] = [
        'Confirme que você é humano',
        'Verificando se você é humano.',
    ]
    MESSAGE: str = 'Not Human detection.'


class AccessDeniedDetection(BotDetection):
    """Detecta páginas de acesso negado ("Access Denied" / "Access denied").

    Example:
        >>> handler = ChromeWebDriverHandler(
        ...     bot_detection_methods=[AccessDeniedDetection]
        ... )
    """

    CONTENT: List[str] = [
        'Access denied.',
        'Access Denied.',
    ]
    MESSAGE: str = 'Access Denied.'
