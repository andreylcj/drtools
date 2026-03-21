

from typing import List, Dict
import logging
from .handler import WebDriverHandler
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.webdriver import WebDriver as SeleniumChromeWebDriver
from webdriver_manager.chrome import ChromeDriverManager
from seleniumwire.webdriver import Chrome as ChromeWebDriver
from fake_useragent import UserAgent


class ChromeWebDriverHandler(WebDriverHandler):
    """Handler de automação web baseado no Google Chrome via Selenium Wire.

    Implementação concreta de :class:`~drtools.web_automation.driver_handler.handler.WebDriverHandler`
    para Chrome, com suporte a:

    - Modo headless (sem interface gráfica).
    - Controle de carregamento de imagens e JavaScript.
    - Prevenção de detecção de automação (desabilita flags do Chromium, randomiza User-Agent).
    - Proxy via ``seleniumwire_options``.
    - Configuração de pasta de download.
    - Preferências customizadas do Chrome.

    Example:
        >>> from drtools.web_automation.driver_handler.chrome import ChromeWebDriverHandler
        >>> from drtools.web_automation.bot_detection import GoogleBotDetection, BlockDetection
        ...
        >>> handler = ChromeWebDriverHandler(
        ...     bot_detection_methods=[GoogleBotDetection, BlockDetection],
        ...     bot_detection_max_retries=3,
        ... )
        ...
        >>> # Inicialização básica (com UI, sem imagens, prevenção de bot)
        >>> handler.start(load_images=False, prevent_bot_detection=True)
        ...
        >>> # Inicialização headless para ambientes de servidor
        >>> handler.start(
        ...     remove_ui=True,
        ...     load_images=False,
        ...     download_path="/tmp/downloads",
        ... )
        ...
        >>> # Com proxy via seleniumwire
        >>> handler.start(
        ...     seleniumwire_options={
        ...         "proxy": {
        ...             "http": "http://user:pass@proxy.example.com:8080",
        ...             "https": "http://user:pass@proxy.example.com:8080",
        ...             "verify_ssl": False,
        ...         }
        ...     }
        ... )
        ...
        >>> handler.go_to_page("https://www.amazon.com")
        >>> handler.quit()
    """

    def start(
        self,
        options: ChromeOptions=None,
        options_arguments: List[str]=[],
        load_images: bool=False,
        load_js: bool=True,
        remove_ui: bool=False,
        prevent_bot_detection: bool=True,
        warning_logs: bool=True,
        seleniumwire_options: Dict=None,
        executable_path: str=None,
        download_path: str=None,
        language: str='en-US',
        disable_password_save: bool=True,
        custom_prefs: Dict={}
    ) -> None:
        """Inicializa o Chrome WebDriver com as configurações fornecidas.

        Configura opções do Chrome, instala o ChromeDriver via ``webdriver_manager``
        (se ``executable_path`` não for informado), cria o driver Selenium ou
        Selenium Wire (quando ``seleniumwire_options`` é fornecido) e aplica
        as medidas de prevenção de detecção de bot quando solicitado.

        Args:
            options: Instância de :class:`~selenium.webdriver.chrome.options.Options`
                pré-configurada. Se ``None``, uma nova é criada.
            options_arguments: Lista de argumentos adicionais do Chrome
                (e.g. ``["--proxy-server=host:port"]``). Argumentos de
                ``window-size``, ``start-maximized`` e ``lang`` têm defaults
                aplicados automaticamente se ausentes.
            load_images: Se ``True``, carrega imagens. Padrão: ``False`` (imagens
                desabilitadas para maior velocidade).
            load_js: Se ``True`` (padrão), executa JavaScript. Se ``False``, desabilita JS.
            remove_ui: Se ``True``, inicia em modo headless com ``--headless=new``,
                ``--no-sandbox``, ``--disable-gpu`` e ``--mute-audio``.
                Padrão: ``False``.
            prevent_bot_detection: Se ``True`` (padrão), desabilita flags de automação
                do Chromium (``AutomationControlled``), remove o switch
                ``enable-automation``, desabilita a extensão de automação e
                sobrescreve o User-Agent com um valor aleatório de usuário real.
            warning_logs: Se ``True`` (padrão), suprime logs verbosos do Selenium
                e urllib3 abaixo do nível WARNING.
            seleniumwire_options: Dict de opções para o Selenium Wire
                (e.g. proxy, SSL). Se ``None`` ou vazio, usa o Selenium padrão
                sem Selenium Wire.
            executable_path: Caminho para o ``chromedriver`` binário. Se ``None``,
                instala automaticamente via ``ChromeDriverManager``.
            download_path: Caminho absoluto do diretório de download padrão.
                Se ``None``, usa o padrão do Chrome.
            language: Idioma do navegador. Padrão: ``"en-US"``.
            disable_password_save: Se ``True`` (padrão), desabilita o gerenciador
                de senhas e o popup de salvar senha do Chrome.
            custom_prefs: Dict de preferências adicionais do Chrome a mesclar
                após as configurações padrão (tem prioridade sobre os defaults).

        Example:
            >>> handler = ChromeWebDriverHandler()
            ...
            >>> # Headless sem imagens, idioma pt-BR
            >>> handler.start(
            ...     remove_ui=True,
            ...     load_images=False,
            ...     language="pt-BR",
            ...     download_path="/tmp/relatorios",
            ... )
            ...
            >>> # Com argumentos extras e prefs customizadas
            >>> handler.start(
            ...     options_arguments=["--window-size=1280x800"],
            ...     custom_prefs={"download.prompt_for_download": False},
            ... )
        """

        # set options if not provided
        if not options:
            options = ChromeOptions()

        # add options arguments
        has_window_size = False
        has_start_maximized = False
        has_lang = False
        for arg in options_arguments:
            if 'window-size' in arg:
                has_window_size = True
            if 'start-maximized' in arg:
                has_start_maximized = True
            if 'lang=' in arg:
                has_lang = True
            options.add_argument(arg)
        if not has_window_size:
            options.add_argument('--window-size=1920x1080')
        if not has_start_maximized:
            options.add_argument('--start-maximized')
        if not has_lang:
            options.add_argument(f'--lang={language}')

        # Set chrome prefs
        chrome_prefs = {
            "profile.default_content_setting_values": {},
        }

        # not load images
        if not load_images:
            chrome_prefs['profile.default_content_setting_values']['images'] = 2

        # not load js
        if not load_js:
            chrome_prefs['profile.default_content_setting_values']['javascript'] = 2

        # set download path
        if download_path:
            chrome_prefs['download.default_directory'] = download_path

        # disable password save
        if disable_password_save:
            chrome_prefs['credentials_enable_service'] = False
            chrome_prefs['profile.password_manager_enabled'] = False
            chrome_prefs['profile.password_manager_leak_detection'] = False

        # Set Experimental Options
        previous_prefs = options.experimental_options.get("prefs", {})
        prefs = {**chrome_prefs, **previous_prefs}
        self.download_path = prefs.get("download.default_directory", download_path)
        prefs = {**prefs, **custom_prefs}
        self.LOGGER.info(f"Chrome Prefs: {prefs}")
        options.experimental_options["prefs"] = prefs

        # Remove UI
        if remove_ui:
            remove_ui_args = [
                '--headless=new',
                '--no-sandbox',
                '--disable-gpu',
                '--mute-audio',
            ]
            for remove_ui_arg in remove_ui_args:
                if remove_ui_arg not in options_arguments:
                    options.add_argument(remove_ui_arg)

        # Prevent bot detection
        if prevent_bot_detection:
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)

        # Only display possible problems
        if warning_logs:
            logging.getLogger('selenium.webdriver.remote.remote_connection') \
                .setLevel(logging.WARNING)
            logging.getLogger('urllib3.connectionpool') \
                .setLevel(logging.WARNING)

        if not executable_path:
            executable_path = ChromeDriverManager().install()

        # Start selenium wire instance only if seleniumwire_options is not empty
        if not seleniumwire_options:
            driver = SeleniumChromeWebDriver(options, ChromeService(executable_path))
        else:
            driver = ChromeWebDriver(options, ChromeService(executable_path), seleniumwire_options=seleniumwire_options)

        # Prevent bot detection
        if prevent_bot_detection:
            user_agent = UserAgent(browsers="chrome", os="windows", platforms="pc")
            user_agent = user_agent.getChrome['useragent']
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": user_agent})

        self.set_driver(driver)

    def quit(self) -> None:
        """Encerra o Chrome WebDriver com tratamento de erros.

        Chama ``driver.quit()`` e loga qualquer exceção sem propagá-la,
        garantindo que a automação não quebre no encerramento.

        Example:
            >>> handler.quit()
        """
        try:
            self.driver.quit()
        except Exception as exc:
            self.LOGGER.error(f"{exc}")
