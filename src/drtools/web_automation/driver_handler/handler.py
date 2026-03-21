

from typing import List, Any, Callable, Tuple
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
import time
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
from drtools.logging import Logger, FormatterOptions
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from ..bot_detection import BotDetection
from ..exceptions import (
    BotDetectionError,
    BotDetectionMaxRetriesError
)
from .config import (
    DEFAULT_BOT_DETECTION_METHODS,
    DEFAULT_BOT_DETECTION_MAX_RETRIES,
    DEFAULT_BOT_DETECTION_RETRY_WAIT_TIME,
    DEFAULT_BOT_DETECTION_WAIT_FOR_PRESENCE_DELAY,
)
from drtools.utils import retry, remove_break_line


class WebDriverHandler:
    """Wrapper de alto nível sobre o Selenium :class:`~selenium.webdriver.remote.webdriver.WebDriver`.

    Centraliza as operações mais comuns de automação web: navegação, busca de
    elementos, clique, scroll, upload de arquivo, login automático e detecção de bot.

    Subclasses devem implementar :meth:`start` e :meth:`quit` para inicializar
    e encerrar o driver específico (Chrome, Firefox, etc.).

    Args:
        driver: Instância de :class:`~selenium.webdriver.remote.webdriver.WebDriver`.
            Pode ser ``None`` e definido depois via :meth:`set_driver`.
        LOGGER: Logger da drtools. Usa logger padrão ``"WebDriverHandler"`` se não informado.
        bot_detection_methods: Lista de classes de detecção de bot a verificar
            após cada carregamento de página.
        bot_detection_max_retries: Número máximo de retries ao detectar bot.
        bot_detection_retry_wait_time: Segundos de espera entre retries.
        bot_detection_wait_for_presence_delay: Timeout para aguardar o elemento
            de detecção de bot.

    Example:
        >>> from drtools.web_automation.driver_handler.chrome import ChromeWebDriverHandler
        >>> from drtools.web_automation.bot_detection import GoogleBotDetection
        ...
        >>> handler = ChromeWebDriverHandler(
        ...     bot_detection_methods=[GoogleBotDetection],
        ...     bot_detection_max_retries=5,
        ... )
        >>> handler.start(remove_ui=True, load_images=False)
        >>> handler.go_to_page("https://www.google.com")
        >>> el = handler.find_element('//input[@name="q"]')
        >>> el.send_keys("drtools python")
        >>> handler.find_then_click('//input[@name="btnK"]')
        >>> handler.quit()
    """

    driver: WebDriver = None

    def __init__(
        self,
        driver: WebDriver=None,
        LOGGER: Logger=None,
        bot_detection_methods: List[BotDetection]=DEFAULT_BOT_DETECTION_METHODS,
        bot_detection_max_retries: int=DEFAULT_BOT_DETECTION_MAX_RETRIES,
        bot_detection_retry_wait_time: int=DEFAULT_BOT_DETECTION_RETRY_WAIT_TIME,
        bot_detection_wait_for_presence_delay: int=DEFAULT_BOT_DETECTION_WAIT_FOR_PRESENCE_DELAY,
    ) -> None:
        if not LOGGER:
            LOGGER = Logger(
                name="WebDriverHandler",
                formatter_options=FormatterOptions(include_datetime=True, include_logger_name=True, include_level_name=True),
                default_start=False
            )
        self.set_driver(driver)
        self.set_logger(LOGGER)
        self.bot_detection_methods = bot_detection_methods
        self.bot_detection_max_retries = bot_detection_max_retries
        self.bot_detection_retry_wait_time = bot_detection_retry_wait_time
        self.bot_detection_wait_for_presence_delay = bot_detection_wait_for_presence_delay
        self.download_path = None

    def set_bot_detection_config(
        self,
        bot_detection_methods: List[BotDetection],
        bot_detection_max_retries: int=DEFAULT_BOT_DETECTION_MAX_RETRIES,
        bot_detection_retry_wait_time: int=DEFAULT_BOT_DETECTION_RETRY_WAIT_TIME,
        bot_detection_wait_for_presence_delay: int=DEFAULT_BOT_DETECTION_WAIT_FOR_PRESENCE_DELAY,
    ) -> None:
        """Reconfigura os parâmetros de detecção de bot em tempo de execução.

        Args:
            bot_detection_methods: Nova lista de detectores de bot.
            bot_detection_max_retries: Novo número máximo de retries.
            bot_detection_retry_wait_time: Novo tempo de espera entre retries (segundos).
            bot_detection_wait_for_presence_delay: Novo timeout de detecção (segundos).

        Example:
            >>> from drtools.web_automation.bot_detection import BlockDetection, HumanDetection
            >>> handler.set_bot_detection_config(
            ...     bot_detection_methods=[BlockDetection, HumanDetection],
            ...     bot_detection_max_retries=5,
            ...     bot_detection_retry_wait_time=30,
            ... )
        """
        self.bot_detection_methods = bot_detection_methods
        self.bot_detection_max_retries = bot_detection_max_retries
        self.bot_detection_retry_wait_time = bot_detection_retry_wait_time
        self.bot_detection_wait_for_presence_delay = bot_detection_wait_for_presence_delay

    def start(self, *args, **kwargs) -> None:
        """Inicializa o WebDriver. **Deve ser implementado pelas subclasses.**

        Raises:
            NotImplementedError: Sempre que não for sobrescrito.
        """
        raise NotImplementedError

    def quit(self, *args, **kwargs) -> None:
        """Encerra o WebDriver. **Deve ser implementado pelas subclasses.**

        Raises:
            NotImplementedError: Sempre que não for sobrescrito.
        """
        raise NotImplementedError

    def set_download_path(self, download_path: str) -> None:
        """Define o caminho de download padrão do handler.

        Args:
            download_path: Caminho absoluto do diretório de downloads.
        """
        self.download_path = download_path

    def set_logger(self, LOGGER: Logger) -> None:
        """Substitui o logger do handler.

        Args:
            LOGGER: Novo logger da drtools.
        """
        self.LOGGER = LOGGER

    def set_actions(self, actions: ActionChains) -> None:
        """Define a instância de :class:`~selenium.webdriver.common.action_chains.ActionChains`.

        Chamado automaticamente em :meth:`set_driver`.

        Args:
            actions: Instância de ActionChains ligada ao driver atual.
        """
        self._actions = actions

    def set_driver(self, driver: WebDriver) -> None:
        """Define o driver Selenium e recria as ActionChains associadas.

        Args:
            driver: Nova instância de WebDriver (pode ser ``None``).
        """
        self._driver = driver
        self.set_actions(ActionChains(self.get_driver()))

    def add_bot_detection_method(self, bot_detection: BotDetection) -> None:
        """Adiciona um detector de bot à lista, se ainda não estiver presente.

        Args:
            bot_detection: Classe de detecção a adicionar.
        """
        if bot_detection not in self.bot_detection_methods:
            self.bot_detection_methods.append(bot_detection)

    def get_actions(self) -> ActionChains:
        """Retorna a instância de :class:`~selenium.webdriver.common.action_chains.ActionChains`."""
        return self._actions

    def get_driver(self) -> WebDriver:
        """Retorna a instância do WebDriver atual."""
        return self._driver

    @property
    def actions(self) -> ActionChains:
        """Atalho de propriedade para :meth:`get_actions`."""
        return self._actions

    @property
    def driver(self) -> WebDriver:
        """Atalho de propriedade para :meth:`get_driver`."""
        return self._driver

    @property
    def driver_windows_count(self) -> int:
        """Número de janelas/abas abertas no driver atual."""
        return len(self.driver.window_handles)

    def hover_element(
        self,
        xpath: str,
        wait: bool=False,
        delay: int=5,
        raise_exception: bool=True
    ):
        """Move o cursor do mouse sobre o elemento identificado pelo XPath.

        Args:
            xpath: XPath do elemento alvo.
            wait: Se ``True``, aguarda a presença do elemento antes de mover.
            delay: Timeout em segundos para aguardar o elemento (usado quando ``wait=True``).
            raise_exception: Se ``True``, propaga exceção caso o elemento não seja encontrado.

        Example:
            >>> handler.hover_element('//button[@id="menu"]', wait=True, delay=3)
        """
        if wait:
            element = self.wait_for_element_presence_located_by_xpath(xpath, delay, raise_exception)
        else:
            element = self.find_element(xpath)
        self.actions.move_to_element(element).perform()

    def retry(
        self,
        func,
        max_tries=5,
        wait_time: float=1,
        raise_exception: bool=False,
        return_if_not_success: Any=None,
    ) -> Tuple:
        """Executa ``func`` com retry automático usando o logger do handler.

        Wrapper sobre :func:`drtools.utils.retry` que injeta automaticamente
        o ``LOGGER`` do handler.

        Args:
            func: Callable a ser executado.
            max_tries: Número máximo de tentativas. Padrão: ``5``.
            wait_time: Segundos de espera entre tentativas. Padrão: ``1``.
            raise_exception: Se ``True``, levanta a última exceção após esgotar tentativas.
            return_if_not_success: Valor retornado quando todas as tentativas falham
                e ``raise_exception=False``.

        Returns:
            Tupla ``(resultado, última_exceção)``.

        Example:
            >>> result, exc = handler.retry(
            ...     func=lambda: handler.find_element('//div[@id="result"]'),
            ...     max_tries=3,
            ...     wait_time=2,
            ... )
        """
        return retry(
            func=func,
            max_tries=max_tries,
            wait_time=wait_time,
            raise_exception=raise_exception,
            return_if_not_success=return_if_not_success,
            LOGGER=self.LOGGER
        )

    def find_then_click(
        self,
        query: str,
        reference_el: WebElement=None,
        by: By=By.XPATH,
        raise_exception: bool=False,
        js: bool=False,
        wait_for_el: bool=None,
        wait_time: int=5,
    ) -> WebElement:
        """Encontra um elemento e clica nele.

        Por padrão aguarda a presença do elemento quando ``reference_el`` não
        é fornecido.

        Args:
            query: XPath (ou seletor conforme ``by``) do elemento.
            reference_el: Elemento de referência para busca relativa. Se ``None``,
                busca no documento inteiro.
            by: Estratégia de busca. Padrão: ``By.XPATH``.
            raise_exception: Se ``True``, propaga exceção caso o elemento não seja encontrado.
            js: Se ``True``, realiza o clique via JavaScript.
            wait_for_el: Se ``True``, aguarda a presença antes de clicar.
                Padrão: ``True`` quando ``reference_el`` é ``None``.
            wait_time: Timeout em segundos para aguardar o elemento.

        Returns:
            O elemento clicado, ou ``None`` se não encontrado.

        Example:
            >>> handler.find_then_click('//button[@type="submit"]')
            >>> handler.find_then_click('//a[text()="Login"]', wait_for_el=True, wait_time=10)
        """
        if wait_for_el is None:
            wait_for_el = reference_el is None
        if wait_for_el:
            element = self.wait_for_element_presence_located_by_xpath(query, wait_time, raise_exception)
        else:
            element = self.find_element(query, reference_el, by, raise_exception)
        if element:
            self.click(element, js)
            return element

    def auto_login(
        self,
        username: str,
        password: str
    ) -> WebDriver:
        """Realiza login automático em formulários HTML padrão.

        Busca o campo de senha pelo tipo ``password``, preenche-o, encontra
        o campo de usuário como o input não-oculto imediatamente anterior,
        preenche-o e clica no botão de submit do formulário.

        Args:
            username: Nome de usuário a preencher.
            password: Senha a preencher.

        Returns:
            O :class:`~selenium.webdriver.remote.webdriver.WebDriver` ativo.

        Example:
            >>> handler.go_to_page("https://example.com/login")
            >>> handler.auto_login("admin", "secret123")
        """
        password_xpath = "//input[@type='password']"
        password_input = self.driver.find_element(by=By.XPATH, value=password_xpath)
        password_input.clear()
        password_input.send_keys(password)

        username_xpath = "preceding::input[not(@type='hidden')][1]"
        username_input = password_input.find_element(by=By.XPATH, value=username_xpath)
        username_input.clear()
        username_input.send_keys(username)

        form_element = password_input.find_element(by=By.XPATH, value=".//ancestor::form")
        submit_button = form_element.find_element(by=By.XPATH, value=".//*[@type='submit']")
        self.click(submit_button)

    def _wait_executor(
        self,
        method: Callable,
        delay: int=5,
        raise_exception: bool=True
    ) -> WebElement:
        """Executa um callable que representa uma condição esperada do Selenium.

        Captura :class:`~selenium.common.exceptions.TimeoutException` e,
        dependendo de ``raise_exception``, relança ou retorna ``None``.

        Args:
            method: Callable sem argumentos que encapsula o ``WebDriverWait``.
            delay: Timeout em segundos (usado na mensagem de erro).
            raise_exception: Se ``True``, relança :class:`TimeoutException` com
                mensagem descritiva. Se ``False``, retorna ``None``.

        Returns:
            O :class:`~selenium.webdriver.remote.webelement.WebElement` encontrado,
            ou ``None`` se timeout e ``raise_exception=False``.
        """
        response = None
        try:
            response = method()
            self.LOGGER.debug("Expected conditions were satisfied!")
        except TimeoutException:
            if raise_exception:
                raise TimeoutException(f"Expected conditions took too much time ({delay:,}s)!")
            response = None
        return response

    def _wait_for_element_by_xpath(
        self,
        xpath: str,
        delay: int,
        expected_conditions: str,
        action: str,
    ) -> Callable:
        """Constrói um callable que aguarda uma condição do Selenium para um XPath.

        Args:
            xpath: XPath do elemento a aguardar.
            delay: Timeout em segundos.
            expected_conditions: Nome do método de
                :mod:`selenium.webdriver.support.expected_conditions`
                (e.g. ``"presence_of_element_located"``, ``"element_to_be_clickable"``).
            action: Método a chamar no elemento após encontrá-lo
                (e.g. ``"click"``, ``"is_displayed"``). Vazio para nenhuma ação.

        Returns:
            Callable que, quando invocado, executa a espera e retorna o elemento.
        """
        self.LOGGER.debug(f"Wait {expected_conditions} for element {xpath} during {delay}s. Acion: {action}")
        def _func(*args, **kwargs) -> WebElement:
            element = WebDriverWait(self.driver, delay).until(
                getattr(EC, expected_conditions)((By.XPATH, xpath))
            )
            if action:
                getattr(element, action)()
            return element
        return _func

    def wait_for_element_to_be_clickable_by_xpath(
        self,
        xpath: str,
        delay: int=5,
        raise_exception: bool=True
    ) -> WebElement:
        """Aguarda um elemento ser clicável pelo XPath e clica nele.

        Args:
            xpath: XPath do elemento.
            delay: Timeout em segundos. Padrão: ``5``.
            raise_exception: Se ``True``, lança exceção ao atingir timeout.

        Returns:
            O elemento clicado, ou ``None`` se timeout e ``raise_exception=False``.

        Example:
            >>> btn = handler.wait_for_element_to_be_clickable_by_xpath(
            ...     '//button[@id="submit"]', delay=10
            ... )
        """
        return self._wait_executor(
            self._wait_for_element_by_xpath(
                xpath,
                delay,
                "element_to_be_clickable",
                "click"
            ),
            delay,
            raise_exception
        )

    def wait_for_element_presence_located_by_xpath(
        self,
        xpath: str,
        delay: int=5,
        raise_exception: bool=True
    ) -> WebElement:
        """Aguarda a presença de um elemento no DOM pelo XPath.

        Usa ``presence_of_element_located`` do Selenium e chama ``is_displayed``
        no elemento encontrado.

        Args:
            xpath: XPath do elemento a aguardar.
            delay: Timeout em segundos. Padrão: ``5``.
            raise_exception: Se ``True``, lança exceção ao atingir timeout.

        Returns:
            O elemento encontrado, ou ``None`` se timeout e ``raise_exception=False``.

        Example:
            >>> modal = handler.wait_for_element_presence_located_by_xpath(
            ...     '//div[@class="modal"]', delay=15
            ... )
        """
        return self._wait_executor(
            self._wait_for_element_by_xpath(
                xpath,
                delay,
                "presence_of_element_located",
                "is_displayed"
            ),
            delay,
            raise_exception
        )

    def close_current_tab(self) -> None:
        """Fecha a aba atual do navegador.

        Example:
            >>> handler.open_tab("https://example.com")
            >>> handler.go_to_tab(1)
            >>> # ... faz algo na nova aba ...
            >>> handler.close_current_tab()
            >>> handler.go_to_tab(0)
        """
        self.driver.close()

    def go_to_tab(self, tab_index: int=0) -> None:
        """Muda o foco para a aba indicada pelo índice.

        Args:
            tab_index: Índice da aba (0 = primeira, -1 = última). Padrão: ``0``.

        Example:
            >>> handler.open_tab("https://example.com")
            >>> handler.go_to_tab(1)   # vai para a nova aba
            >>> handler.go_to_tab(0)   # volta para a primeira aba
        """
        self.driver.switch_to.window(self.driver.window_handles[tab_index])

    def open_tab(self, url: str='') -> None:
        """Abre uma nova aba no navegador via JavaScript e muda o foco para ela.

        Args:
            url: URL a abrir na nova aba. Vazio para aba em branco.

        Example:
            >>> handler.open_tab("https://example.com")
            >>> # foco agora está na nova aba
            >>> handler.go_to_tab(0)  # volta para a primeira aba
        """
        try:
            self.driver.execute_script(f'window.open("{url}","_blank");')
            self.go_to_tab(-1)
        except:
            self.go_to_tab(0)
            self.driver.execute_script(f'window.open("{url}","_blank");')
            self.go_to_tab(-1)

    def go_to_page(self, url: str) -> None:
        """Navega para a URL informada com retry automático em caso de detecção de bot.

        Após cada carregamento, verifica todos os detectores de
        :attr:`bot_detection_methods`. Se detectado, aguarda
        :attr:`bot_detection_retry_wait_time` segundos e tenta novamente.
        Após :attr:`bot_detection_max_retries` tentativas sem sucesso, lança
        :class:`~drtools.web_automation.exceptions.BotDetectionMaxRetriesError`.

        Args:
            url: URL de destino.

        Raises:
            BotDetectionMaxRetriesError: Quando o número máximo de retries é atingido.

        Example:
            >>> handler.go_to_page("https://www.amazon.com/dp/B08N5WRWNW")
            >>> title = handler.find_element('//span[@id="productTitle"]')
            >>> print(title.text)
        """
        self.LOGGER.debug(f"Go to page: {url}...")
        retries = -1
        while True:
            self.driver.get(url)
            bot_detection_located = False
            for bot_detection in self.bot_detection_methods:
                try:
                    bot_detection.detect(self)
                except BotDetectionError as exc:
                    bot_detection_located = True
                    break
            if not bot_detection_located:
                break
            retries += 1
            if retries >= self.bot_detection_max_retries:
                raise BotDetectionMaxRetriesError(f"Bot detection retry attempts reach maximum {self.bot_detection_max_retries:,}.")
            self.LOGGER.debug(f'Sleeping for {self.bot_detection_retry_wait_time:,}s. Retry: {retries:,}...')
            time.sleep(self.bot_detection_retry_wait_time)
            self.LOGGER.debug(f'Sleeping for {self.bot_detection_retry_wait_time:,}s. Retry: {retries:,}... Done!')
        self.LOGGER.debug(f"Go to page: {url}... Done!")

    def find_element(
        self,
        query: str,
        reference_el: WebElement=None,
        by: By=By.XPATH,
        raise_exception: bool=False,
    ) -> WebElement:
        """Busca um único elemento na página.

        Se ``reference_el`` for fornecido, a busca parte desse elemento.
        Caso contrário, busca no documento completo via ``self.driver``.

        Args:
            query: XPath ou seletor de busca.
            reference_el: Elemento pai para busca relativa. Padrão: ``None``.
            by: Estratégia de busca do Selenium. Padrão: ``By.XPATH``.
            raise_exception: Se ``True``, propaga exceção em caso de falha.
                Se ``False``, retorna ``None``.

        Returns:
            :class:`~selenium.webdriver.remote.webelement.WebElement` encontrado,
            ou ``None`` se não encontrado e ``raise_exception=False``.

        Raises:
            Exception: Se nem ``driver`` nem ``reference_el`` forem fornecidos.

        Example:
            >>> title = handler.find_element('//h1[@class="product-title"]')
            >>> if title:
            ...     print(title.text)

            >>> # Busca relativa a um elemento pai
            >>> card = handler.find_element('//div[@class="product-card"][1]')
            >>> price = handler.find_element('.//span[@class="price"]', reference_el=card)
        """
        result = None
        try:
            if reference_el is not None:
                result = reference_el.find_element(by, value=query)
            elif self.driver is not None:
                result = self.driver.find_element(by, value=query)
            else:
                raise Exception('Provide "driver" or "reference_el".')
        except Exception as exc:
            exc_msg = str(exc.msg)
            self.LOGGER.debug(remove_break_line(exc_msg))
            if raise_exception:
                raise exc
            result = None
        return result

    def find_element_on_shadow_root(
        self,
        parent_shadow_query: str,
        query: str,
        by: By=By.CSS_SELECTOR,
        parent_shadow_reference_el: WebElement=None,
        parent_shadow_by: By=By.XPATH,
        raise_exception: bool=False,
    ) -> WebElement:
        """Busca um elemento dentro de um Shadow DOM.

        Primeiro localiza o elemento pai que contém o shadow root, depois
        busca dentro dele.

        Args:
            parent_shadow_query: Query do elemento pai que possui shadow root.
            query: Query do elemento dentro do shadow root.
            by: Estratégia de busca dentro do shadow root. Padrão: ``By.CSS_SELECTOR``.
            parent_shadow_reference_el: Elemento de referência para busca do pai.
            parent_shadow_by: Estratégia de busca do pai. Padrão: ``By.XPATH``.
            raise_exception: Se ``True``, propaga exceção em caso de falha.

        Returns:
            :class:`~selenium.webdriver.remote.webelement.WebElement` encontrado,
            ou ``None``.

        Example:
            >>> el = handler.find_element_on_shadow_root(
            ...     parent_shadow_query='//my-component',
            ...     query='button.submit',
            ... )
        """
        parent_shadow = self.find_element(parent_shadow_query, parent_shadow_reference_el, parent_shadow_by, raise_exception)
        element = self.find_element(query, parent_shadow.shadow_root, by, raise_exception)
        return element

    def find_element_on_shadow_root_then_click(
        self,
        parent_shadow_query: str,
        query: str,
        by: By=By.CSS_SELECTOR,
        parent_shadow_reference_el: WebElement=None,
        parent_shadow_by: By=By.XPATH,
        raise_exception: bool=False,
        js: bool=False,
        wait_for_el: bool=False,
        wait_time: int=5,
    ) -> None:
        """Encontra um elemento dentro de um Shadow DOM e clica nele.

        Args:
            parent_shadow_query: Query do elemento pai com shadow root.
            query: Query do elemento dentro do shadow root.
            by: Estratégia de busca dentro do shadow root. Padrão: ``By.CSS_SELECTOR``.
            parent_shadow_reference_el: Referência para busca do pai.
            parent_shadow_by: Estratégia de busca do pai. Padrão: ``By.XPATH``.
            raise_exception: Se ``True``, propaga exceção em caso de falha.
            js: Se ``True``, realiza clique via JavaScript.
            wait_for_el: Se ``True``, aguarda a presença do pai antes de continuar.
            wait_time: Timeout em segundos para aguardar o pai.

        Example:
            >>> handler.find_element_on_shadow_root_then_click(
            ...     parent_shadow_query='//my-modal',
            ...     query='button.confirm',
            ...     wait_for_el=True,
            ...     wait_time=10,
            ... )
        """
        if wait_for_el:
            element = self.wait_for_element_presence_located_by_xpath(parent_shadow_query, wait_time)
            element = self.find_element_on_shadow_root(
                parent_shadow_query,
                query,
                by,
                parent_shadow_reference_el,
                parent_shadow_by,
                raise_exception
            )
        else:
            element = self.find_element_on_shadow_root(
                parent_shadow_query,
                query,
                by,
                parent_shadow_reference_el,
                parent_shadow_by,
                raise_exception
            )
        self.click(element, js)

    def find_elements(
        self,
        query: str,
        reference_el: any=None,
        by: By=By.XPATH
    ) -> List[WebElement]:
        """Busca múltiplos elementos na página.

        Args:
            query: XPath ou seletor de busca.
            reference_el: Elemento pai para busca relativa. Padrão: ``None``.
            by: Estratégia de busca. Padrão: ``By.XPATH``.

        Returns:
            Lista de :class:`~selenium.webdriver.remote.webelement.WebElement`,
            ou ``None`` em caso de erro.

        Example:
            >>> items = handler.find_elements('//ul[@class="results"]//li')
            >>> for item in items:
            ...     print(item.text)

            >>> # Busca relativa
            >>> table = handler.find_element('//table[@id="data"]')
            >>> rows = handler.find_elements('.//tr', reference_el=table)
        """
        result = None
        try:
            if reference_el is not None:
                result = reference_el.find_elements(by, value=query)
            elif self.driver is not None:
                result = self.driver.find_elements(by, value=query)
            else:
                raise Exception('Provide "driver" or "reference_el".')
        except Exception as exc:
            exc_msg = str(exc.msg)
            self.LOGGER.debug(remove_break_line(exc_msg))
            result = None
        return result

    def perform_click(
        self,
        element: WebElement,
        js: bool=False
    ) -> None:
        """Executa um clique direto em um elemento, via Selenium ou JavaScript.

        Args:
            element: Elemento alvo do clique.
            js: Se ``True``, usa ``driver.execute_script("arguments[0].click()", element)``.
                Requer ``self.driver`` definido. Padrão: ``False``.

        Raises:
            AssertionError: Se ``js=True`` e ``self.driver`` for ``None``.

        Example:
            >>> btn = handler.find_element('//button[@id="ok"]')
            >>> handler.perform_click(btn)
            >>> handler.perform_click(btn, js=True)  # clique via JS
        """
        if js:
            assert self.driver is not None, \
                'When "js" == True, you need must set "driver" static attribute'
        if js:
            self.driver.execute_script("arguments[0].click();", element)
        else:
            element.click()

    def click(
        self,
        element: WebElement,
        js: bool=False,
        max_tries: int=2,
        js_when_exaust: bool=True,
        wait_time: int=1,
    ) -> None:
        """Clica em um elemento com retry automático e fallback via JavaScript.

        Tenta clicar via Selenium até ``max_tries`` vezes. Se todas as tentativas
        falharem e ``js_when_exaust=True``, realiza uma última tentativa via JavaScript.

        Args:
            element: Elemento a clicar.
            js: Se ``True``, usa JavaScript desde a primeira tentativa.
            max_tries: Número máximo de tentativas Selenium. Padrão: ``2``.
            js_when_exaust: Se ``True`` (padrão), tenta clique JS ao esgotar retries.
            wait_time: Segundos de espera entre tentativas. Padrão: ``1``.

        Raises:
            Exception: Se todas as tentativas (incluindo JS) falharem.

        Example:
            >>> el = handler.find_element('//button[@id="submit"]')
            >>> handler.click(el)
            >>> handler.click(el, js=True)  # força clique via JS
            >>> handler.click(el, max_tries=5, wait_time=2)
        """
        success, last_exception = self.retry(
            func=lambda: self.perform_click(element, js=js),
            max_tries=max_tries,
            wait_time=wait_time,
            return_if_not_success=False
        )
        if success is False:
            if js_when_exaust \
            and self.driver is not None \
            and not js:
                self.perform_click(element, js=True)
            else:
                raise last_exception

    def set_input_range_value(
        self,
        input_slide_element: WebElement,
        value: int
    ) -> None:
        """Define o valor de um input do tipo ``range`` (slider) via teclas de seta.

        Calcula a diferença entre o valor atual e o desejado e envia o número
        correto de teclas ``LEFT`` ou ``RIGHT``.

        Args:
            input_slide_element: Elemento ``<input type="range">`` alvo.
            value: Valor desejado para o slider.

        Example:
            >>> slider = handler.find_element('//input[@type="range"][@id="price-filter"]')
            >>> handler.set_input_range_value(slider, 50)
        """
        curr_val = int(input_slide_element.get_attribute('value'))
        is_right_key = value > curr_val
        if is_right_key:
            max_val = int(input_slide_element.get_attribute('max'))
            max_val = max(value, max_val)
            for i in range(max_val - curr_val):
                input_slide_element.send_keys(Keys.RIGHT)
        else:
            min_val = int(input_slide_element.get_attribute('min'))
            min_val = min(value, min_val)
            for i in range(curr_val, min_val, -1):
                input_slide_element.send_keys(Keys.LEFT)

    def window_scroll(
        self,
        scroll_to: int
    ):
        """Faz scroll da janela para uma posição Y absoluta via JavaScript.

        Args:
            scroll_to: Posição Y em pixels para rolar.

        Example:
            >>> handler.window_scroll(500)   # rola 500px para baixo
            >>> handler.window_scroll(0)     # volta ao topo
        """
        self.driver.execute_script(f"window.scrollTo(0, {scroll_to})")

    def window_infinite_scroll(
        self,
        scroll_pause_time: float=1.5,
        timeout: float=30,
        scroll_middle_action: Callable=None
    ):
        """Rola a janela até o final da página de forma contínua (infinite scroll).

        Continua rolando enquanto o ``scrollHeight`` da página aumentar ou até
        o timeout ser atingido.

        Args:
            scroll_pause_time: Segundos de pausa após cada scroll. Padrão: ``1.5``.
            timeout: Tempo máximo em segundos. Padrão: ``30``.
            scroll_middle_action: Callable opcional chamado após cada pausa
                (útil para interações intermediárias).

        Raises:
            Exception: Se o scroll não terminar dentro do timeout.

        Example:
            >>> # Scroll simples até o fim
            >>> handler.window_infinite_scroll()

            >>> # Com ação intermediária (ex: fechar popups)
            >>> handler.window_infinite_scroll(
            ...     scroll_middle_action=lambda: handler.find_then_click('//button[@class="close-popup"]')
            ... )
        """
        last_height = int(self.driver.execute_script("return document.body.scrollHeight"))
        started_at = datetime.now()
        while True:
            self.window_scroll(last_height)
            time.sleep(scroll_pause_time)
            if scroll_middle_action:
                scroll_middle_action()
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            duration = (datetime.now() - started_at).total_seconds()
            if duration > timeout:
                raise Exception(f"Scroll not finish {round(duration, 2)}s.")

    def perform_scroll(
        self,
        element: WebElement,
        height: float,
    ):
        """Rola o scroll interno de um elemento para uma posição Y via JavaScript.

        Útil para elementos com overflow (listas, tabelas, painéis com scroll próprio).

        Args:
            element: Elemento com scroll interno.
            height: Posição ``scrollTop`` desejada em pixels.

        Example:
            >>> panel = handler.find_element('//div[@class="scrollable-panel"]')
            >>> handler.perform_scroll(panel, 300)
        """
        self.driver.execute_script("arguments[0].scrollTop = arguments[1]", element, height)

    def scroll_into_view(
        self,
        element: WebElement
    ):
        """Rola a página até que o elemento fique visível na viewport.

        Args:
            element: Elemento a tornar visível.

        Example:
            >>> el = handler.find_element('//footer')
            >>> handler.scroll_into_view(el)
        """
        self.driver.execute_script("arguments[0].scrollIntoView(true);", element)

    def scroll_to_start_of_page(self) -> None:
        """Rola a página até o topo usando ``Ctrl+Home``.

        Example:
            >>> handler.scroll_to_start_of_page()
        """
        self.find_element('//body').send_keys(Keys.CONTROL + Keys.HOME)

    def scroll_to_end_of_page(self) -> None:
        """Rola a página até o final usando ``Ctrl+End``.

        Example:
            >>> handler.scroll_to_end_of_page()
        """
        self.find_element('//body').send_keys(Keys.CONTROL + Keys.END)

    def perform_infinite_scroll(
        self,
        element: WebElement,
        scroll_pause_time: float=1.5,
        timeout: float=30,
        scroll_middle_action: Callable=None
    ):
        """Rola o scroll interno de um elemento até o final (infinite scroll em container).

        Análogo a :meth:`window_infinite_scroll` mas para elementos com scroll próprio
        (e.g. listas com lazy loading dentro de um div).

        Args:
            element: Elemento com scroll interno.
            scroll_pause_time: Segundos de pausa após cada scroll. Padrão: ``1.5``.
            timeout: Tempo máximo em segundos. Padrão: ``30``.
            scroll_middle_action: Callable opcional chamado após cada pausa.

        Raises:
            Exception: Se o scroll não terminar dentro do timeout.

        Example:
            >>> feed = handler.find_element('//div[@class="feed"]')
            >>> handler.perform_infinite_scroll(feed, scroll_pause_time=2, timeout=60)
        """
        last_height = int(element.get_attribute("scrollHeight"))
        started_at = datetime.now()
        while True:
            self.perform_scroll(element, last_height)
            time.sleep(scroll_pause_time)
            if scroll_middle_action:
                scroll_middle_action()
            new_height = int(element.get_attribute("scrollHeight"))
            if new_height == last_height:
                break
            last_height = new_height
            duration = (datetime.now() - started_at).total_seconds()
            if duration > timeout:
                raise Exception(f"Scroll not finish {round(duration, 2)}s.")

    def refresh_page(self, simple: bool=True):
        """Recarrega a página atual.

        Args:
            simple: Se ``True`` (padrão), usa ``driver.refresh()``.
                Se ``False``, envia ``Ctrl+R`` via teclado no body.

        Example:
            >>> handler.refresh_page()
            >>> handler.refresh_page(simple=False)  # via teclado
        """
        self.LOGGER.debug("Refreshing page...")
        if simple:
            self.driver.refresh()
        else:
            self.find_element('//body').send_keys(Keys.COMMAND + 'r')
        self.LOGGER.debug("Refreshing page... Done!")
