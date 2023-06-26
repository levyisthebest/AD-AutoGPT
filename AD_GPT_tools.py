
import os
import json
# from autogpt.logs import logger
from duckduckgo_search import ddg
import requests
from bs4 import BeautifulSoup
from requests import Response
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from requests.compat import urljoin
import pandas as pd
import geopandas as gpd
from urllib import request
from geotext import GeoText

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from shapely.geometry import Point, Polygon
import descartes
from dateparser.search import search_dates


def google_search(query: str, num_results: int = 8) -> str:
    """Return the results of a Google search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    search_results = []
    if not query:
        print(1)
        return json.dumps(search_results)

    results = ddg(query, max_results=num_results)
    print(results)
    # return results
    if not results:
        return json.dumps(search_results)

    for j in results:
        search_results.append(j)

    results = json.dumps(search_results, ensure_ascii=False, indent=4)
    # print(results)
    return safe_google_results(results)


def safe_google_results(results: str ) -> str:
    """
        Return the results of a google search in a safe format.

    Args:
        results (str | list): The search results.

    Returns:
        str: The results of the search.
    """
    if isinstance(results, list):
        safe_message = json.dumps(
            [result.encode("utf-8", "ignore") for result in results]
        )
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    return safe_message

def execute_python_file(filename: str) -> str:
    """Execute a Python file in a Docker container and return the output

    Args:
        filename (str): The name of the file to execute

    Returns:
        str: The output of the file
    """
    # logger.info(f"Executing file '{filename}'")
    print("Executing file '{filename}'")
    # f"python {filename}"
    cmd = "python "+filename
    os.system(cmd)

def scrape_text(url: str, news_url = 'bbc')->str:
    """Scrape text from a webpage

    Args:
        url (str): The URL to scrape text from

    Returns:
        str: The scraped text
    """
    # session = requests.Session()
    # session.trust_env = False
    # page = session.get(url,verify= False)
    page = requests.get(url,verify= False)
    soup = BeautifulSoup(page.content, features="html.parser")
    print(soup)
    time_flag = []
    
    if(news_url == 'bbc' or  news_url == 'NIA'):
        time_flag = 'time'
        time_element = soup.find(time_flag)
        datetime_value = str(time_element.get('datetime')).split("T")[0]
        
    if(news_url == 'AA'):
        time_flag = "metaDate"
        time_element = soup.find("div", class_=time_flag)
        datetime_value = str(time_element.contents).split("['\r\n  ")[1].split("\n")[0]
    
    if(news_url == 'ARUK'):
        time_element = soup.find("meta", {"property":"article:published_time"}, content=True)
        datetime_value = str(time_element["content"]).split("T")[0]
        
    if(news_url == 'Mayo'):
        time_element = soup.find("span", class_="moddate")
        datetime_value = str(time_element.contents[0])
    
    if(news_url == 'AE'):
        time_element = soup.find("div", class_="fl-module fl-module-rich-text fl-node-5e6b8729db02f news_date")
        datetime_value = time_element.contents
        
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    
    return text, datetime_value



def format_hyperlinks(hyperlinks: list[tuple[str, str]]) -> list[str]:
    """Format hyperlinks to be displayed to the user

    Args:
        hyperlinks (List[Tuple[str, str]]): The hyperlinks to format

    Returns:
        List[str]: The formatted hyperlinks
    """
    return [f"{link_text} ({link_url})" for link_text, link_url in hyperlinks]

def extract_hyperlinks(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    """Extract hyperlinks from a BeautifulSoup object

    Args:
        soup (BeautifulSoup): The BeautifulSoup object
        base_url (str): The base URL

    Returns:
        List[Tuple[str, str]]: The extracted hyperlinks
    """
    return [
        (link.text, urljoin(base_url, link["href"]))
        for link in soup.find_all("a", href=True)
    ]

def scrape_links(url: str) -> str | list[str]:
    """Scrape links from a webpage

    Args:
        url (str): The URL to scrape links from

    Returns:
       str | list[str]: The scraped links
    """
    page = requests.get(url, verify = False)
    soup = BeautifulSoup(page.content, "html.parser")
    
    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup, url)

    return format_hyperlinks(hyperlinks)




def scrape_place_text(text):
    geolocator = Nominatim(user_agent="Icarus",timeout=2)
    places = GeoText(text)
    cities = list(set(list(places.cities)))
    cities_out = [] 
    for city in cities:
        try:
            location = geolocator.geocode(city)
            if location:
                cities_out.append(city)
        except GeocoderTimedOut as e:
            print(str(city)+' is not a city')
    return cities


# t, time = scrape_text('https://www.alzheimersresearchuk.org/deep-brain-stimulation-restores-memory-in-mice-with-alzheimers-symptoms/','AE')
# print(time)