

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
from drtools.utils import retry


class WebDriverHandler:
    
    driver: WebDriver = None # Must be set
    
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
        self.bot_detection_methods = bot_detection_methods
        self.bot_detection_max_retries = bot_detection_max_retries
        self.bot_detection_retry_wait_time = bot_detection_retry_wait_time
        self.bot_detection_wait_for_presence_delay = bot_detection_wait_for_presence_delay

    def start(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    def quit(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    def set_download_path(self, download_path: str) -> None:
        self.download_path = download_path
    
    def set_logger(self, LOGGER: Logger) -> None:
        self.LOGGER = LOGGER
        
    def set_actions(self, actions: ActionChains) -> None:
        self._actions = actions
    
    def set_driver(self, driver: WebDriver) -> None:
        self._driver = driver
        self.set_actions(ActionChains(self.get_driver()))
        
    def add_bot_detection_method(self, bot_detection: BotDetection) -> None:
        if bot_detection not in self.bot_detection_methods:
            self.bot_detection_methods.append(bot_detection)
        
    def get_actions(self) -> ActionChains:
        return self._actions
        
    def get_driver(self) -> WebDriver:
        return self._driver
    
    @property
    def actions(self) -> ActionChains:
        return self._actions
    
    @property
    def driver(self) -> WebDriver:
        return self._driver
    
    @property
    def driver_windows_count(self) -> int:
        return len(self.driver.window_handles)
    
    def hover_element(
        self,
        xpath: str,
        wait: bool=False,
        delay: int=5,
        raise_exception: bool=True
    ):
        # get element
        if wait:
            element = self.wait_for_element_presence_located_by_xpath(xpath, delay, raise_exception)
        else:
            element = self.find_element(xpath)
        # hover el
        self.actions.move_to_element(element).perform()
    
    def retry(
        self,
        func, 
        max_tries=5,
        wait_time: float=1,
        raise_exception: bool=False,
        return_if_not_success: Any=None,
    ) -> Tuple:
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
        wait_for_el: bool=False,
        wait_time: int=5,
    ) -> WebElement:
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
        """Perform automatic login

        Parameters
        ----------
        username : str
            The username that will be placed on login form
        password : str
            The password that will be placed on password input

        Returns
        -------
        WebDriver
            Selenium WebDriver
        """
        # Find a password input field and enter the specified password string
        password_xpath = "//input[@type='password']"
        # password_input = wait_for_element(driver, EC.presence_of_element_located((By.XPATH, password_xpath)))
        password_input = self.driver.find_element(by=By.XPATH, value=password_xpath)
        # driver.execute_script(f"arguments[0].value = \"{password}\";", password_input)
        password_input.clear()
        password_input.send_keys(password)

        # Find a visible input field preceding out password field and enter the specified username
        # username_abs_xpath = f"{password_xpath}/preceding::input[not(@type='hidden')][1]"
        # username_input = wait_for_element(driver, EC.presence_of_element_located((By.XPATH, username_abs_xpath)))
        username_xpath = "preceding::input[not(@type='hidden')][1]"
        username_input = password_input.find_element(by=By.XPATH, value=username_xpath)
        # driver.execute_script(f"arguments[0].value = \"{username}\";", username_input)
        username_input.clear()
        username_input.send_keys(username)

        # Find the form element enclosing our password field
        form_element = password_input.find_element(by=By.XPATH, value=".//ancestor::form")

        # Find the form's submit element and click it
        submit_button = form_element.find_element(by=By.XPATH, value=".//*[@type='submit']")
        self.click(submit_button)
    
    def _wait_executor(
        self, 
        method: Callable,
        delay: int=5,
        raise_exception: bool=True
    ) -> WebElement:
        """Wait for element until expected conditions were satisfied or delay time.

        Raises
        ------
        TimeoutException
            If conditions were not satisfied whithin delay time.
        """
        response = None
        # started_at = datetime.now()
        try:
            response = method()
            # duration = (datetime.now() - started_at).total_seconds()
            # duration = round(duration, 2)
            # self.LOGGER.debug(f"Expected conditions were satisfied in {duration}!")
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
        """Close current tab of Browser;

        Parameters
        ----------
        driver : WebDriver
            Selenium WebDriver

        Returns
        -------
        WebDriver
            Selenium WebDriver
        """
        self.driver.close()
    
    def go_to_tab(self, tab_index: int=0) -> None:
        """Go to tab by index

        Parameters
        ----------
        driver : WebDriver
            Selenium WebDriver
        tab_index : int, optional
            Index of tab, from 0 to number of opened tabs
            minus 1, by default 0

        Returns
        -------
        WebDriver
            Selenium WebDriver
        """
        
        self.driver.switch_to.window(self.driver.window_handles[tab_index])
    
    def open_tab(self, url: str='') -> None:
        """Open new tab using JavaScript.

        Parameters
        ----------
        driver : WebDriver
            Selenium WebDrive
        url : str
            Open tab in the desired url

        Returns
        -------
        WebDriver
            Selenium WebDriver
        """
        
        try:
            self.driver.execute_script(f'window.open("{url}","_blank");')
            self.go_to_tab(-1)
        except:
            self.go_to_tab(0)
            self.driver.execute_script(f'window.open("{url}","_blank");')
            self.go_to_tab(-1)

    def go_to_page(self, url: str) -> None:
        """Open page by url

        Parameters
        ----------
        driver : WebDriver
            Selenium WebDriver
        url : str
            The desired URL page.
        """
        self.LOGGER.debug(f"Go to page: {url}...")
        retries = -1
        while True:
            
            self.driver.get(url)
            
            # Verify for bot detection
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
            
            # After wait, Verify for bot detection again. If not find break
            # bot_detection_located = False
            # for bot_detection in self.bot_detection_methods:
            #     try:
            #         bot_detection.detect(self)
            #     except BotDetectionError as exc:
            #         bot_detection_located = True
            #         break
            # if not bot_detection_located:
            #     break
            
        self.LOGGER.debug(f"Go to page: {url}... Done!")
        
    def find_element(
        self,
        query: str, 
        reference_el: WebElement=None, 
        by: By=By.XPATH,
        raise_exception: bool=False,
    ) -> WebElement:
        """Find element on page.

        Parameters
        ----------
        query : str
            The query string, often XPATH query string
        reference_el : WebElement, optional
            The reference element which query will be started
            from, by default None
        by : By, optional
            Search by option, by default By.XPATH

        Returns
        -------
        WebElement
            If find some element, returns the result of 
            search, else returns **None**

        Raises
        ------
        Exception
            If "driver" and "reference_el" is **None**
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
            self.LOGGER.warning(exc_msg.replace("\n", " <BR> "))
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
        """Find elements on page.

        Parameters
        ----------
        query : str
            The query string, often XPATH query string
        reference_el : WebElement, optional
            The reference element which query will be started
            from, by default None
        by : By, optional
            Search by option, by default By.XPATH

        Returns
        -------
        List[WebElement]
            If find elements, returns the result of 
            search, else returns **None**

        Raises
        ------
        Exception
            If "driver" and "reference_el" is **None**
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
            self.LOGGER.warning(exc_msg.replace("\n", " <BR> "))
            result = None
        return result
    
    def perform_click(
        self,
        element: WebElement,
        js: bool=False
    ) -> None:
        """Perform Selenium click or Javascript click

        Parameters
        ----------
        element : WebElement
            The element which will be clicked
        js : bool, optional
            Perform click using Javascript, by default False
            
        Returns
        -------
        None
            **None** is returned
            
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
        """Click action with error handler

        Parameters
        ----------
        element : WebElement
            The element which will be clicked
        js : bool, optional
            Perform click using Javascript, by default True
        retry : int, optional
            Number of times the action will be attempted, by default 5
        js_when_exaust : bool, optional
            When click is not being performed by 
            Javascript ("js" == False), if True, after total number of 
            attempts, the click will try to be performed
            by Javascript, by default True

        Raises
        ------
        Exception
            When click action can not be sucessfuly executed.
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
        """Change value of slide input

        Parameters
        ----------
        input_slide_element : WebElement
            Slide input element
        value : int
            Desired value to be placed on input

        Returns
        -------
        WebDriver
            Selenium WebDriver
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
        self.driver.execute_script(f"window.scrollTo(0, {scroll_to})")
    
    def window_infinite_scroll(
        self,
        scroll_pause_time: float=1.5,
        timeout: float=30,
    ):
        # Get scroll height
        last_height = int(self.driver.execute_script("return document.body.scrollHeight"))
        started_at = datetime.now()
        while True:
            # Scroll down to bottom
            self.window_scroll(last_height)
            # Wait to load page
            time.sleep(scroll_pause_time)

            # Calculate new scroll height and compare with last scroll height
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
        self.driver.execute_script("arguments[0].scrollTop = arguments[1]", element, height)

    def scroll_into_view(
        self,
        element: WebElement
    ):
        self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
        
    def scroll_to_start_of_page(self) -> None:
        self.find_element('//body').send_keys(Keys.CONTROL + Keys.HOME)
        
    def scroll_to_end_of_page(self) -> None:
        self.find_element('//body').send_keys(Keys.CONTROL + Keys.END)

    def perform_infinite_scroll(
        self,
        element: WebElement,
        scroll_pause_time: float=1.5,
        timeout: float=30,
    ):
        # Get scroll height
        last_height = int(element.get_attribute("scrollHeight"))
        started_at = datetime.now()
        while True:
            # Scroll down to bottom
            self.perform_scroll(element, last_height)
            # Wait to load page
            time.sleep(scroll_pause_time)

            # Calculate new scroll height and compare with last scroll height
            new_height = int(element.get_attribute("scrollHeight"))
            if new_height == last_height:
                break
            last_height = new_height
            duration = (datetime.now() - started_at).total_seconds()
            
            if duration > timeout:
                raise Exception(f"Scroll not finish {round(duration, 2)}s.")
            
    def refresh_page(self, simple: bool=True):
        self.LOGGER.debug("Refreshing page...")
        if simple:
            self.driver.refresh()
        else:
            self.find_element('//body').send_keys(Keys.COMMAND + 'r')
        self.LOGGER.debug("Refreshing page... Done!")