# Import required libraries
import os  # For file system operations
import json  # For JSON handling
from duckduckgo_search import ddg  # DuckDuckGo search API
import requests  # For making HTTP requests
from bs4 import BeautifulSoup  # For parsing HTML content
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting graphs
import pandas as pd  # Data manipulation
import geopandas as gpd  # Geographic data operations
from urllib import request  # URL handling
from geotext import GeoText  # For extracting place names from text
import re  # Regular expressions for text matching
from geopy.geocoders import Nominatim  # For geocoding (getting coordinates)
from geopy.exc import GeocoderTimedOut  # Handling timeout errors in geocoding
from shapely.geometry import Point  # For handling point-based geometries
import descartes  # For plotting shapely objects on maps
from dateparser.search import search_dates  # For extracting dates from text
from datetime import date, timedelta  # For handling date and time operations
import spacy  # NLP library
import nltk  # Natural Language Toolkit
import gensim  # Topic modeling
from nltk.corpus import wordnet as wn  # WordNet corpus for lexical relations
from nltk.stem.wordnet import WordNetLemmatizer  # For word lemmatization
from spacy.lang.en import English  # English tokenizer from spaCy
from gensim.utils import simple_preprocess  # Simple preprocessing of text
from gensim.models import CoherenceModel  # For measuring topic coherence
from gensim import corpora  # Dictionary and corpus management
import pyLDAvis.gensim  # Visualization of LDA models
import matplotlib as mpl  # Additional customization for plots
import matplotlib.colors as mcolors  # Handle color mapping

# Load spaCy English model
spacy.load('en_core_web_sm')

# Load NLTK stop words
from nltk.corpus import stopwords

# Extend the list of stop words
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 
                   'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 
                   'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather',
                   'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 
                   'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 
                   'come'])

# Set up HTTP proxies (if needed)
proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'  # Use HTTP proxy for HTTPS
}

def find_files(path, filename):
    """Find all occurrences of a specific file in a directory."""
    results = []
    for root, _, files in os.walk(path):
        for name in files:
            if name == filename + '.txt':
                results.append(os.path.join(root, name))
    return results

def get_city_info(save_path):
    """Extract city information from saved files and plot them on a map."""
    pwd = os.getcwd()
    places_all = []
    geolocator = Nominatim(user_agent="Icarus", timeout=2)
    files = find_files(pwd + '/workplace/', 'places')
    
    for p in files:
        name = p.split('\\')[-3].split("/")[-1]
        with open(p, "r", encoding='utf8') as f:
            for place in f.readlines():
                location = geolocator.geocode(place)
                places_all.append([place.strip(), name, location.address, 
                                   (location.latitude, location.longitude)])
    
    # Create a DataFrame to store city information
    df = pd.DataFrame(places_all, columns=['City Name', 'News_Source', 'Country', 'Coordinates'])    
    df.to_csv(save_path + '/geo_information.csv', index=None)  # Save to CSV

    # Create a GeoDataFrame for plotting
    geometry = [Point(x[1], x[0]) for x in df['Coordinates']]
    crs = {'init': 'epsg:4326'}
    geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    
    # Load a world map shapefile
    countries_map = gpd.read_file(pwd + '/world_map.shp')

    # Plot the map with the cities
    fig, ax = plt.subplots(figsize=(16, 16))
    countries_map.plot(ax=ax, alpha=0.4, color='grey')
    geo_df['geometry'].plot(ax=ax, markersize=30, color='#FF9800', marker='^', alpha=0.5)
    
    plt.title("Where Latest Alzheimer's Disease News Happen", fontsize=24, fontstyle='italic')
    plt.savefig(save_path + '/Places.jpg', dpi=300)  # Save the map plot

def get_lemma(word):
    """Get the lemma (base form) of a word using WordNet."""
    lemma = wn.morphy(word)
    return word if lemma is None else lemma

def tokenize(text):
    """Tokenize and preprocess the text."""
    parser = English()  # Initialize the English tokenizer
    tokens = parser(text)
    return [token.lower_ for token in tokens if not token.orth_.isspace()]

def prepare_text_for_lda(text):
    """Prepare text for LDA topic modeling."""
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4 and token not in stop_words]
    return [get_lemma(token) for token in tokens]

def get_time_difference(date_txt, news_type='bbc'):
    """Calculate the time difference between a news date and today."""
    time_now = datetime.datetime.now()
    if news_type in ['bbc', 'NIA']:
        date_desired = datetime.datetime.strptime(date_txt, '%Y-%m-%d')
    elif news_type in ['AA', 'Mayo']:
        date_desired = datetime.datetime.strptime(date_txt, '%B %d, %Y')
    return max(0, (time_now - date_desired).days)

def get_summary_period(path_dir, save_dir):
    """Generate and save a summary of the news articles."""
    files = find_files(path_dir, 'dates')
    month_num = 12  # Number of months to analyze
    counts = np.zeros(month_num)  # Array to store news counts per month

    # Process each date file and update counts
    for date_f in files:
        with open(date_f, "r", encoding='utf8') as f:
            date_txt = f.readline().strip()
        summary_f = date_f.replace('dates', 'summary')
        with open(summary_f, "r", encoding='utf8') as f:
            summary_txt = f.readline().strip()
        
        time_index = get_time_difference(date_txt) // 30  # Group by month
        counts[min(time_index, month_num - 1)] += 1

    # Plot the distribution of news articles over time
    plt.figure(figsize=(12, 10))
    plt.bar(range(month_num), counts, color='#63b2ee', width=1)
    plt.xticks(range(month_num), ['2023-5', '2023-4', '2023-3', '2023-2', '2023-1', 
                                  '2022-12', '2022-11', '2022-10', '2022-9', 
                                  '2022-8', '2022-7', '2022-6'], rotation=30)
    plt.xlabel('Timeline (month)')
    plt.ylabel('News Count')
    plt.title('News Distribution Over the Past Year', fontsize=24, fontstyle='italic')
    plt.savefig(save_dir + 'news_distribution_last_year.jpg', dpi=300)
    plt.close()

def scrape_text(url, news_url='bbc'):
    """Scrape the text content and metadata from a news article."""
    page = requests.get(url, verify=False, proxies=proxies)
    soup = BeautifulSoup(page.content, "html.parser")
    news_title = soup.find("title").text  # Extract the news title

    if news_url == 'bbc':
        datetime_value = soup.find("time")['datetime'].split("T")[0]
    elif news_url == 'CNN':
        datetime_value = soup.find("div", class_="timestamp").text.strip()
    # Add more conditions for different sources as needed

    # Extract the main text content
    for script in soup(["script", "style"]):
        script.extract()
    text = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())
    
    return text, datetime_value, news_title

# Example usage: Scrape CNN news articles
if __name__ == '__main__':
    url = "https://edition.cnn.com/search?q=climate+change&size=10"
    cnn_articles = newspaper.build(url, memoize_articles=False)
    for article in cnn_articles.articles:
        print(article.url)
