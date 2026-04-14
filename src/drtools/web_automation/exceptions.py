

class BotDetectionError(Exception):
    """Levantada quando uma verificação de detecção de bot é identificada na página.

    Capturada internamente por :meth:`~drtools.web_automation.driver_handler.handler.WebDriverHandler.go_to_page`
    para acionar a lógica de retry antes de repassar ao chamador.

    Example:
        >>> from drtools.web_automation.exceptions import BotDetectionError
        >>> raise BotDetectionError("Google bot detection.")
    """
    pass


class BotDetectionMaxRetriesError(Exception):
    """Levantada quando o número máximo de tentativas de contornar detecção de bot é atingido.

    Propagada por :meth:`~drtools.web_automation.driver_handler.handler.WebDriverHandler.go_to_page`
    quando todas as tentativas de retry se esgotam sem sucesso.

    Example:
        >>> from drtools.web_automation.exceptions import BotDetectionMaxRetriesError
        >>> raise BotDetectionMaxRetriesError("Bot detection retry attempts reach maximum 3.")
    """
    pass
