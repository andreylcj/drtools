

from typing import List, Dict
import logging
from .handler import WebDriverHandler
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from seleniumwire.webdriver import Chrome as ChromeWebDriver
from fake_useragent import UserAgent


class ChromeWebDriverHandler(WebDriverHandler):

    def start(
        self,
        options: ChromeOptions=None,
        options_arguments: List[str]=[],
        # proxy_server: str=None,
        load_images: bool=False,
        load_js: bool=True,
        remove_ui: bool=False,
        prevent_bot_detection: bool=True,
        warning_logs: bool=True,
        seleniumwire_options: Dict=None,
        executable_path: str=None,
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
        for arg in options_arguments:
            options.add_argument(arg)

        # handle proxy_server
        # if proxy_server:
        #     options.add_argument(f'--proxy-server={proxy_server}')

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

        # Set Experimental Options
        previous_prefs = options.experimental_options.get("prefs", {})
        prefs = {**chrome_prefs, **previous_prefs}
        self.downloads_path = prefs.get("download.default_directory", "~/Downloads")
        options.experimental_options["prefs"] = prefs

        # Remove UI
        if remove_ui:
            remove_ui_args = [
                '--headless',
                'start-maximized',
                'window-size=1920x1080',
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

        # Initialize driver
        driver = ChromeWebDriver(options=options, service=ChromeService(executable_path), seleniumwire_options=seleniumwire_options)
        
        # Prevent bot detection
        if prevent_bot_detection:
            user_agent = UserAgent()
            user_agent = user_agent.random
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": user_agent})
        
        self.set_driver(driver)
    
    def quit(self) -> None:
        try:
            self.driver.quit()
        except Exception as exc:
            self.LOGGER.error(f"{exc}")