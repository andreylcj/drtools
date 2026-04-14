

from typing import List


DEFAULT_BOT_DETECTION_METHODS: List = []
"""Lista padrão de métodos de detecção de bot.

Vazia por padrão — nenhum detector ativo. Passe instâncias de subclasses
de :class:`~drtools.web_automation.bot_detection.BotDetection` para ativar
a verificação automática durante :meth:`~drtools.web_automation.driver_handler.handler.WebDriverHandler.go_to_page`.

Example:
    >>> from drtools.web_automation.bot_detection import GoogleBotDetection, BlockDetection
    >>> methods = [GoogleBotDetection, BlockDetection]
"""

DEFAULT_BOT_DETECTION_MAX_RETRIES: int = 3
"""Número máximo de tentativas de retry ao detectar bot. Padrão: ``3``."""

DEFAULT_BOT_DETECTION_RETRY_WAIT_TIME: int = 20
"""Tempo de espera em segundos entre retries de detecção de bot. Padrão: ``20`` segundos."""

DEFAULT_BOT_DETECTION_WAIT_FOR_PRESENCE_DELAY: int = 1
"""Timeout em segundos para aguardar o elemento de detecção de bot na página. Padrão: ``1`` segundo."""
