

from typing import Dict
from ...main.resource import SingletonResource
from drtools.extensions.adspower.local_api_handler import AdspowerHandler
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


class BaseAdspowerResource(SingletonResource):
    """Singleton resource managing an Adspower browser profile lifecycle.

    Handles browser open/close via the Adspower LocalAPI and connects Selenium
    to the launched Chrome instance.

    Class attributes:
        PROFILE_ID: Adspower browser profile ID. Must be set by subclasses.

    Raises:
        Exception: On init if PROFILE_ID is not set.
    """

    NAME: str = "Adspower Base Resource"
    PROFILE_ID: str = None
    PROFILE_NAME: str = None

    def __init__(
        self,
        conf: Dict=None,
        LOGGER = None
    ):
        super().__init__(conf, LOGGER)
        if not self.PROFILE_ID and not self.PROFILE_NAME:
            raise Exception("Static attribute PROFILE_ID or PROFILE_NAME must be set.")
        self.browser = None
        self.driver = None
        self.close_before_open = self.context.conf.get('close_before_open', False) is True
        self._adspower_handler = None
    
    def instantiate_adspower_handler(self):
        """Create and validate an AdspowerHandler, checking that the service is available.

        Stores the handler in self._adspower_handler.

        Raises:
            Exception: If the Adspower service is not available.
        """
        ads_power_handler = AdspowerHandler(LOGGER=self.LOGGER)
        ads_power_status = ads_power_handler.status()
        if ads_power_status and ads_power_status['code'] != 0:
            raise Exception("Ads Power is not available")
        self._adspower_handler = ads_power_handler
        return self._adspower_handler
    
    def get_adspower_handler(self):
        """Return the cached AdspowerHandler, creating it on first access."""
        if not self._adspower_handler:
            self.instantiate_adspower_handler()
        return self._adspower_handler
    
    def start_browser(self):
        """Open the browser profile and initialize the Selenium WebDriver.

        If close_before_open is True (set via conf['close_before_open']),
        any already-active browser for this profile is closed first.

        Populates self.browser and self.driver after a successful open.
        """
        ads_power_handler = self.get_adspower_handler()
        
        if self.close_before_open and self.PROFILE_ID:
            resp = ads_power_handler.check_single_browser_status(params={'user_id': self.PROFILE_ID})
            browser_already_open = resp['data']['status'] == 'Active'
            if browser_already_open:
                try:
                    self.LOGGER.debug('Try close browser before open...')
                    ads_power_handler.close_browser_v2(post_data={'profile_id': self.PROFILE_ID})
                    self.LOGGER.debug('Try close browser before open... Done!')
                except Exception as exc:
                    self.LOGGER.warning(f'Exception when try to close browser: {exc}')
        
        if self.PROFILE_NAME:
            self.LOGGER.debug(f'Opening browser {self.PROFILE_NAME}...')
            self.browser = ads_power_handler.open_browser_v2_by_name(name=self.PROFILE_NAME, case='exact')
            # self.LOGGER.debug('profiles: ' + str(self.browser))
            self.LOGGER.debug(f'Opening browser {self.PROFILE_NAME}... Done!')
        else:
            self.LOGGER.debug(f'Opening browser {self.PROFILE_ID}...')
            self.browser = ads_power_handler.open_browser_v2(post_data={'profile_id': self.PROFILE_ID})
            self.LOGGER.debug(f'Opening browser {self.PROFILE_ID}... Done!')
        
        chrome_driver = self.browser["data"]["webdriver"]
        service = Service(executable_path=chrome_driver)
        chrome_options = Options()
        chrome_options.add_experimental_option("debuggerAddress", self.browser["data"]["ws"]["selenium"])
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
                
    def close_browser(self):
        """Close the browser profile via the Adspower API.

        No-op if no driver is currently active (self.driver is None).
        """
        if self.driver:
            ads_power_handler = self.get_adspower_handler()
            self.LOGGER.debug(f'Closing browser {self.PROFILE_ID}...')
            ads_power_handler.close_browser_v2(post_data={'profile_id': self.PROFILE_ID})
            self.LOGGER.debug(f'Closing browser {self.PROFILE_ID}... Done!')