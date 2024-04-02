

from .exceptions import BotDetectionError
from typing import List


class BotDetection:
    
    CONTENT: List[str] = None
    MESSAGE: str = 'Bot detection.'
    
    @classmethod
    def build_pattern(cls) -> str:
        return f'//*[' + ' or '.join([f'contains(., "{s}")' for s in cls.CONTENT]) + ']'
    
    @classmethod
    def detect(
        cls,
        web_drive_handler,
        raise_exception: bool=True
    ) -> None:
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
    
    CONTENT: List[str] = [
        "Nossos sistemas detectaram tráfego incomum na sua rede de computadores",
    ]
    MESSAGE: str = 'Google bot detection.'


class BlockDetection(BotDetection):
    
    CONTENT: List[str] = [
        "Sorry, you have been blocked",
    ]
    MESSAGE: str = 'Block detected.'


class HumanDetection(BotDetection):
    
    CONTENT: List[str] = [
        'Confirme que você é humano',
        'Verificando se você é humano.',
    ]
    MESSAGE: str = 'Not Human detection.'


class AccessDeniedDetection(BotDetection):
    
    CONTENT: List[str] = [
        'Access denied.',
        'Access Denied.',
    ]
    MESSAGE: str = 'Access Denied.'