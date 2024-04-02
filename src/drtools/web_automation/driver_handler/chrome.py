

from typing import List, Dict
import logging
from .handler import WebDriverHandler
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from seleniumwire.webdriver import Chrome as ChromeWebDriver


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
        seleniumwire_options: Dict=None
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
                '--start-maximized',
                '--headless',
                '--no-sandbox',
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

        # Initialize driver
        driver = ChromeWebDriver(
            options=options, 
            service=ChromeService(ChromeDriverManager().install()),
            seleniumwire_options=seleniumwire_options,
        )
        
        # Prevent bot detection
        if prevent_bot_detection:
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36'})
        
        self.set_driver(driver)
    
    def quit(self) -> None:
        try:
            self.driver.quit()
        except Exception as exc:
            self.LOGGER.error(f"{exc}")