""" 
This module was created to handle
Driver initialization.

"""


from typing import Union, List
from selenium import webdriver
import logging
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.utils import ChromeType
from selenium.webdriver.chrome.webdriver import WebDriver


GOOGLE = 'GOOGLE'
CHROMIUM = 'CHROMIUM'
BRAVE = 'BRAVE'
MSEDGE = 'MSEDGE'


def chrome_start(
    chrome_type: Union[
        GOOGLE, CHROMIUM, BRAVE, MSEDGE
    ]=GOOGLE,
    chrome_options_arguments: List[str]=[],
    warning_logs: bool=True,
    load_images: bool=False,
    load_js: bool=True
) -> WebDriver:
    """Start Selenium Chrome Driver.
    
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
    
    Parameters
    ----------
    chrome_type : Union[GOOGLE, CHROMIUM, BRAVE, MSEDGE]
        The chrome type will be loaded, by default GOOGLE
    chrome_options_arguments : List[str]
        A list of chrome options arguments, by default []
    warning_logs : bool
        Display logs of selenium webdriver and urllib3 in
        warning level, by default True
    load_images : bool
        If images of page should be loaded, by default False    
    load_js : bool
        If Javascript of page should be loaded, by default True
        
    Returns
    -------
    WebDriver
        Selenium WebDriver
    """
    
    # Only display possible problems
    if warning_logs:
        logging.getLogger('selenium.webdriver.remote.remote_connection') \
            .setLevel(logging.WARNING)
        logging.getLogger('urllib3.connectionpool') \
            .setLevel(logging.WARNING)

    chrome_options = webdriver.ChromeOptions()

    # chrome options arguments
    for arg in chrome_options_arguments:
        chrome_options.add_argument(arg)

    ### This blocks images and javascript requests
    chrome_prefs = { "profile.default_content_setting_values": {} }
    
    if not load_images:
        chrome_prefs['profile.default_content_setting_values']['images'] = 2
    if not load_js:
        chrome_prefs['profile.default_content_setting_values']['javascript'] = 2

    chrome_options.experimental_options["prefs"] = chrome_prefs
    ###

    driver = webdriver.Chrome(
        service=ChromeService(ChromeDriverManager(
            chrome_type=getattr(ChromeType, chrome_type)
            ).install()),
        chrome_options=chrome_options
    )
    
    return driver