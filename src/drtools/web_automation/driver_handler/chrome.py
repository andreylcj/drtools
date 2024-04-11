

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
    ) -> None:
        """Start Selenium Wire Chrome Driver.
        
        Example
        -------
        Examples of Chrome Options Arguments usage:
        
        **Remove UI**
        
        - chrome_options.add_argument("--headless")
        - chrome_options.add_argument("--no-sandbox")
        - chrome_options.add_argument("--mute-audio")    
        
        **Change window size**
        
        - chrome_options.add_argument("--start-maximized")
        - chrome_options.add_argument("--window-size=1920x1080")
        
        **Change default download location**
        
        - chrome_options.add_argument("download.default_directory=C:/Downloads")
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
            # TODO - Set random available window size
            options.add_argument('--window-size=1920x1080')
        if not has_start_maximized:
            options.add_argument('--start-maximized')
        if not has_lang:
            options.add_argument(f'--lang={language}')

        # Set chrome prefs
        chrome_prefs = {
            "profile.default_content_setting_values": {},
            # "download.default_directory" : "./downloads"
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

        # Set Experimental Options
        previous_prefs = options.experimental_options.get("prefs", {})
        prefs = {**chrome_prefs, **previous_prefs}
        self.download_path = prefs.get("download.default_directory", download_path)
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
            # Initialize driver
            driver = SeleniumChromeWebDriver(options, ChromeService(executable_path))
        else:
            # Initialize driver
            driver = ChromeWebDriver(options, ChromeService(executable_path), seleniumwire_options=seleniumwire_options)
        
        # Prevent bot detection
        if prevent_bot_detection:
            user_agent = UserAgent(browsers="chrome", os="windows", platforms="pc")
            user_agent = user_agent.getChrome['useragent']
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": user_agent})
        
        self.set_driver(driver)
    
    def quit(self) -> None:
        try:
            self.driver.quit()
        except Exception as exc:
            self.LOGGER.error(f"{exc}")