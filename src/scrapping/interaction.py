""" 
This module was created to handle
interaction on Web Site page.

"""


from typing import List
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ExpectedConditions
from selenium.common.exceptions import TimeoutException
import logging
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement


def wait_for_element(
    driver: WebDriver, 
    expected_conditions: ExpectedConditions,
    delay: int=5
) -> WebDriver:
    """Wait for element until expected conditions were satisfied or delay time.

    Parameters
    ----------
    driver : WebDriver
        The Selenium WebDriver
    expected_conditions : FunctionType
        Desired expected conditions to be satisfied
    delay : int, optional
        Max waiting time in seconds until conditions 
        were satisfied, by default 5

    Returns
    -------
    WebDriver
        The WebDriver

    Raises
    ------
    TimeoutException
        If conditions were not satisfied whithin delay time.
    """
    
    my_elem = None
    try:
        my_elem = WebDriverWait(driver, delay).until(expected_conditions)
        logging.debug("Expected conditions were satisfied!")
    except TimeoutException:
        raise TimeoutException("Expected conditions took too much time!")
    return my_elem


def close_current_tab(
    driver: WebDriver
) -> WebDriver:
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
    driver.close()
    return driver


def open_tab(
    driver: WebDriver, 
    url: str=''
) -> WebDriver:
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
        driver.execute_script(f'window.open("{url}","_blank");')
        go_to_tab(driver, -1)
    except:
        go_to_tab(driver, 0)
        driver.execute_script(f'window.open("{url}","_blank");')
        go_to_tab(driver, -1)
    return driver


def go_to_tab(
    driver: WebDriver, 
    tab_index: int=0
) -> WebDriver:
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
    
    driver.switch_to.window(driver.window_handles[tab_index])
    return driver


def go_to_page(
    driver: WebDriver, 
    url: str
) -> WebDriver:
    """Open page by url

    Parameters
    ----------
    driver : WebDriver
        Selenium WebDriver
    url : str
        The desired URL page.

    Returns
    -------
    WebDriver
        Selenium WebDriver
    """
    driver.get(url)
    return driver


def find_element(
    query: str, 
    driver: WebDriver=None, 
    reference_el: WebElement=None, 
    by: By=By.XPATH
) -> WebElement:
    """Find element on page.

    Parameters
    ----------
    query : str
        The query string, often XPATH query string
    driver : WebDriver, optional
        Selenium Driver, by default None
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
        elif driver is not None:
            result = driver.find_element(by, value=query)
        else:
            raise Exception('Provide "driver" or "reference_el".')
    except Exception as exc:
        logging.warning(exc.msg)
        result = None
    return result


def find_elements(
    query: str, 
    driver: WebDriver=None, 
    reference_el: any=None, 
    by: By=By.XPATH
) -> List[WebElement]:
    """Find elements on page.

    Parameters
    ----------
    query : str
        The query string, often XPATH query string
    driver : WebDriver, optional
        Selenium Driver, by default None
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
        elif driver is not None:
            result = driver.find_elements(by, value=query)
        else:
            raise Exception('Provide "driver" or "reference_el".')
    except Exception as exc:
        logging.warning(exc.msg)
        result = None
    return result


def perform_click(
    element: WebElement,
    driver: WebDriver=None,
    js: bool=False
) -> None:
    """Perform Selenium click or Javascript click

    Parameters
    ----------
    element : WebElement
        The element which will be clicked
    driver : WebDriver, optional
        Selenium Driver, by default None
    js : bool, optional
        Perform click using Javascript, by default False
        
    Returns
    -------
    None
        **None** is returned
        
    """
    
    if js:
        assert driver is not None, \
            'When "js" == True, you need provide "driver"'
    if js:
        driver.execute_script("arguments[0].click();", element)
    else:
        element.click()


def click(
    element: WebElement, 
    driver: WebDriver=None, 
    js: bool=True, 
    retry: int=5, 
    js_when_exaust: bool=True
) -> None:
    """Click action with error handler

    Parameters
    ----------
    element : WebElement
        The element which will be clicked
    driver : WebDriver, optional
        Selenium Driver, by default None
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
        
    retry_count = 0
    while retry_count < retry:
        try:
            perform_click(element, driver, js)
            retry_count = float('inf')
        except Exception as exc:
            retry_count = retry_count + 1
            msg = exc
            logging.debug(f'retry count: {retry_count}')
            logging.debug(msg)
    if retry_count != float('inf'):
        if js_when_exaust and driver is not None:
            perform_click(element, driver, js=True)
        else:
            raise Exception(msg)


def set_input_range_value(
    input_slide_element: WebElement, 
    driver: WebDriver, 
    value: int
) -> WebDriver:
    """Change value of slide input

    Parameters
    ----------
    input_slide_element : WebElement
        Slide input element
    driver : WebDriver
        Selenium WebDriver
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
    return driver


def get_el_on_el_list(xpath):
    """get xpath value that represent list, yield first el and increase
    by 1 iterator num"""
    pass