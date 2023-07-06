
import os
import json
# from autogpt.logs import logger
from duckduckgo_search import ddg
import requests
from bs4 import BeautifulSoup
from requests import Response
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
# %matplotlib inline
from requests.compat import urljoin
import pandas as pd
import geopandas as gpd
from urllib import request
from geotext import GeoText
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from shapely.geometry import Point, Polygon
import descartes
from dateparser.search import search_dates
from datetime import date, timedelta
import spacy
import nltk
import gensim
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.en import English
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import corpora
import matplotlib.colors as mcolors
import matplotlib as mpl
import pyLDAvis.gensim

from urllib import request

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from shapely.geometry import Point, Polygon
import descartes


spacy.load('en_core_web_sm')

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 
                   'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see',
                   'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 
                   'right', 'line', 'even', 'also', 'may', 'take', 'come'])

proxies={
'http': 'http://127.0.0.1:7890',
'https': 'http://127.0.0.1:7890'  # https -> http
}
def find_files(path, A):
    results = []
    for root, _, files in os.walk(path):
        for name in files:
            if name == A+'.txt':
                results.append(os.path.join(root,name))
    return results

def get_city_info(save_path):
    pwd = os.getcwd()
    places_all = []
    geolocator = Nominatim(user_agent="Icarus",timeout=2)
    files = find_files(pwd+'/workplace/','places')
    for p in files:
        # print(p)
        name = p.split('\\')[-3].split("/")[-1]
        with open(p, "r", encoding='utf8') as f:
            for places in f.readlines():
                location = geolocator.geocode(places)
                places_all.append([places.replace('\n',''),name,location[0],(location.latitude, location.longitude)])
                
    df = pd.DataFrame(places_all, columns=['City Name', 'News_Source','Country','Coordinates'])    
    df.to_csv(save_path+'/geo_information.csv',index=None)   
    geometry = [Point(x[1], x[0]) for x in df['Coordinates']]
    crs = {'init': 'epsg:4326'}
    geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    countries_map =gpd.read_file(pwd + '/world_map.shp')
    f, ax = plt.subplots(figsize=(16, 16))
    countries_map.plot(ax=ax, alpha=0.4, color='grey')
    geo_df.loc[geo_df['News_Source']=='AA','color'] = '#C62828'
    geo_df.loc[geo_df['News_Source']=='Mayo','color'] = '#283593'
    geo_df.loc[geo_df['News_Source']=='bbc','color'] = '#FF9800'
    geo_df.loc[geo_df['News_Source']=='NIA','color'] = '#82B0D2'
    cmap = ['#C62828','#283593','#FF9800','#283593']
    geo_df['geometry'].plot(ax=ax, markersize = 30,color =geo_df['color'],marker = '^', alpha=.5)
    font_dict = dict(fontsize =24, family = 'Times New Roman', style = 'italic')
    plt.title("Where Latest Alzheimer's Disease News Happen",fontdict= font_dict)
    plt.savefig(save_path + '/Places.jpg',dpi = 300)    
    
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def tokenize(text):
    parser = English()
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def prepare_text_for_lda(text):
    en_stop = set(nltk.corpus.stopwords.words('english'))
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

def get_time_difference(date_txt,news_type='bbc')->int: 
    time_now = datetime.datetime.now()
    date_desired = []
    if news_type == 'bbc' or news_type == 'NIA':
        date_desired = datetime.datetime.strptime(date_txt, '%Y-%m-%d')
    if news_type == 'AA' or news_type == 'Mayo':
        # print(len(date_txt.split("-")))
        if(len(date_txt.split("-"))<=1):      
            date_desired = datetime.datetime.strptime(date_txt, '%B %d, %Y')
        else:
            date_desired = datetime.datetime.strptime(date_txt, '%Y-%m-%d')
    # if news_type == 'Mayo':
    #     date_desired = datetime.datetime.strptime(date_txt, '%b. %d, %Y')
    # print(date_desired, news_type)
    day_difference = (time_now-date_desired).days
    if day_difference>=0:
        return day_difference
    else:
        return 0
    
def find_files(path, A):
    results = []
    for root, _, files in os.walk(path):
        for name in files:
            if name == A+'.txt':
                results.append(os.path.join(root,name))
    return results

def get_summary_period(path_dir,save_dir):
    result = [] 
    # text_month = str()
    # text_2month = str()
    # text_6month = str()
    # text_12month = str()
    if os.path.exists(os.getcwd()+'\\workplace\\AA\\'):
        news_path = os.getcwd()+'\\workplace\\AA\\'
        newslist = os.listdir(news_path)
        count=0
        for news in newslist:
            count = int(news.split('_')[1])
            if(not (os.path.exists(news_path+news+'/dates.txt'))):
                print(news)
                # time_now = datetime.datetime.now()
                date_desired1 = (date.today()-timedelta(days=int(365*count/140)))
                # print(date_desired1.strftime('%B %d, %Y'))
                with open(news_path+news+"/dates.txt", "w", encoding='utf8') as f:
                    f.writelines(str(date_desired1)+'\n')   
    files = find_files(path_dir,'dates')
    month_num = 12
    re_l = np.zeros(month_num)    
    list_text_month = []
    X_label = []
    for i in range(0,month_num):
        list_text_month.append(str())
        X_label.append(str(i+1))
    for date_f in files:        
        with open(date_f, "r", encoding='utf8') as f:
            # print(date_f)
            date_txt = f.readline().replace('\n','')
            # print(date_f)
            news_type = date_f.split("\\")[-3].split("/")[-1]
        summary_f = date_f.replace('dates','summary')
        with open(summary_f, "r", encoding='utf8') as f:
            summary_txt = f.readline().replace('\n','')
        day_difference = get_time_difference(date_txt,news_type)
        time_index = int(day_difference/30)
        if time_index >=month_num:
            time_index = month_num-1
        re_l[time_index]+=1
        list_text_month[time_index] = list_text_month[time_index] + summary_txt
    X= np.arange(month_num)    
    plt.figure(figsize=(12,10))
    plt.bar(2.4*X,re_l, color = '#63b2ee', linewidth = 1.5, width=1)
    X1 = ['2023-5','2023-4','2023-3','2023-2','2023-1','2022-12','2022-11','2022-10','2022-9','2022-8','2022-7','2022-6']
    plt.xticks(ticks=2.4*X, labels=X1, rotation = 30, fontsize =20 )
    plt.xlabel('Timeline (month)',fontsize =20 )
    plt.ylabel('News Count',fontsize =20 )
    plt.title('The Number of Relevant News that Happened in the Past Period',fontdict=dict(fontsize =24, family = 'Times New Roman', style = 'italic'))
    plt.savefig(save_dir+'news_distribution_last_year.jpg', dpi = 300)
    plt.close()
    result = list_text_month
    return result

def text_all_lda(text_result,save_path):
    topic_num_all = [4,3,3,4,3,3,3,3,3,3,4,4]
    # topic_num_AA = [4,4,3,4,4,4,4,4,3,4,3,3] 
    # topic_num_bbc = [4,3,3,4,4,3,4,3,3,4,3,3]
    # topic_num_NIA = [4,4,4,3,3,3,4,3,3,4,3,3]
    df_final = []
    K = []
    for i in range(12):
        K.append(str(i+1)+'_Month_Keywords')
    text_all = str()
    for text_data, k,topic_num in zip(text_result, K, topic_num_all):
        if len(text_data)>0:
            text_all = text_all + text_data
            data_words  = prepare_text_for_lda(text_data)
            # print(data_words)
            
            # Build the bigram and trigram models
            bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
            trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN']): ##'VERB', 'ADJ', , 'ADV'
                """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
                texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
                texts = [bigram_mod[doc] for doc in texts]
                texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
                texts_out = []
                nlp = spacy.load("en_core_web_sm")
                #nlp = spacy.load('en', disable=['parser', 'ner'])
                for sent in texts:
                    doc = nlp(" ".join(sent)) 
                    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
                # remove stopwords once more after lemmatization
                texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
                return texts_out
            data_ready = process_words(data_words)  # processed Text Data!
            # Create Dictionary
            id2word = corpora.Dictionary(data_ready)

            # Create Corpus: Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in data_ready]

            # Build LDA model
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=topic_num, 
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=10,
                                                    passes=10,
                                                    alpha='symmetric',
                                                    iterations=100,
                                                    per_word_topics=True)

            from collections import Counter
            topics = lda_model.show_topics(formatted=False)
            data_flat = [w for w_list in data_ready for w in w_list]
            counter = Counter(data_flat)

            out = []
            for i, topic in topics:
                for word, weight in topic:
                    out.append([word, i , weight, counter[word]])

            df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        
            topic_words ={}
            for i in range(0,topic_num):
                topic_words.update(dict(topics[i][1]))
            # print(topic_words)
            np.save(save_path + 'topic_words_in_'+k+'.npy',topic_words)
            df['Time'] = k
            # df.to_csv('D:\\23spring\\AD-GPT\\workplace\\topic_words_in_'+k+'.csv', index =None)
            df_final.append(df)
            # pyLDAvis.enable_notebook()
            vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
            pyLDAvis.save_html(vis, save_path + 'topic_words_in_'+k+'.html')
    df_result = pd.concat(df_final)
    df_result.to_csv(save_path + 'Topics_Trend_All.csv', index =None)
    data_words  = prepare_text_for_lda(text_all)
    # print(data_words)
    num_topics_1 = 5
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    data_ready = process_words(data_words)  # processed Text Data!
    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics_1, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=10,
                                            passes=10,
                                            alpha='symmetric',
                                            iterations=100,
                                            per_word_topics=True)
    from collections import Counter
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in data_ready for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        
    topic_words ={}
    for i in range(0,num_topics_1):
        topic_words.update(dict(topics[i][1]))
    print(topic_words)
    df.to_csv(save_path+'topic_words_in_all_text.csv', index =None)
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    pyLDAvis.save_html(vis,save_path+'topic_words_in_all_text.html')


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
    page = requests.get(url,verify= False, proxies=proxies)
    soup = BeautifulSoup(page.content, features="html.parser")
    # print(soup)
    time_flag = []
    news_title = str()
    if(news_url == 'bbc' or  news_url == 'NIA'):
        time_flag = 'time'
        time_element = soup.find(time_flag)
        # print(time_element)
        datetime_value = str(time_element.get('datetime')).split("T")[0]
        
        
    if(news_url == 'AA'):
        time_flag = "metaDate"
        time_element = soup.find("div", class_=time_flag)
        print(time_element)
        datetime_value = 0
        if(time_element != None):
            if(len(str(time_element.contents).split("['\\r\\n  "))>=2):
                # print(str(time_element.contents).split("['\\r\\n  "))
                datetime_value = str(time_element.contents).split("['\\r\\n  ")[1].split("\\n")[0]
            else:
                datetime_value = 'March 1, 2023'
    
    if(news_url == 'ARUK'):
        time_element = soup.find("meta", {"property":"article:published_time"}, content=True)
        datetime_value = str(time_element["content"]).split("T")[0]
        
    if(news_url == 'Mayo'):
        time_element = soup.find("span", class_="moddate")
        # print(time_element)
        datetime_value = str(time_element.contents[0])
    
    if(news_url == 'AE'):
        time_element = soup.find("div", class_="fl-module fl-module-rich-text fl-node-5e6b8729db02f news_date")
        datetime_value = time_element.contents

    if(news_url == 'CNN'):
        time_element = soup.find("div", class_="timestamp")
        datetime_value = time_element.contents  
        pattern = r"\w+ \d+,\s\d+"
        matches = re.findall(pattern, str(datetime_value))
        datetime_value = str(matches[0])
        news_title = soup.find("title").contents[0] 
    
    if(news_url == 'Fox'):
        time_element = soup.find("div", class_="article-date")
        datetime_value = time_element.contents  
        pattern = r"\w+ \d+,\s\d+"
        matches = re.findall(pattern, str(datetime_value))
        datetime_value = str(matches[0])
        news_title = soup.find("title").contents[0] 
    
    if(news_url == 'Hill'):
        time_element = soup.find("section", class_="submitted-by | header__meta | text-transform-upper text-300 color-light-gray weight-semibold font-base desktop-only")
        datetime_value = time_element.contents  
        pattern = r"\d{1,2}/\d{1,2}/\d{2}"
        # pattern = r"\w+ \d+,\s\d+"
        matches = re.findall(pattern, str(datetime_value))
        datetime_value = str(matches[0])
        news_title = soup.find("title").contents[0] 
    
    if(news_url == 'NPR'):
        time_element = soup.find("span", class_="date")
        datetime_value = time_element.contents[0]  
        news_title = soup.find("div", class_="storytitle").find("h1").contents[0]
        
    
    if(news_url == 'USAToday'):
        time_element = str(soup)
        # print(time_element)
        # datetime_value = time_element.contents[0] 
        pattern = r"\w+ \d+,\s\d+"
        matches = re.findall(pattern, time_element)
        # print(matches)
        datetime_value = str(matches[0])
        news_title = soup.find("title").contents[0] 

    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    
    return text, datetime_value, news_title



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
    page = requests.get(url, verify = False,proxies=proxies)
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

# import urllib3
# import re
# urllib3.disable_warnings()
# t, time, title = scrape_text('https://www.usatoday.com/story/money/2023/06/28/how-to-lower-home-insurance-costs/70359055007/','USAToday')
# print(title)
# # print(str(time).split('\\n')[-2].split(', '))
# # pattern = r"\w+ \d+,\s\d+"
# # matches = re.findall(pattern, str(time))
# print(time)

import requests
from bs4 import BeautifulSoup

# def get_cnn_search_results(query):
#     url = f"https://www.cnn.com/search/?q={query}&size=10&type=article"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")
#     results = []

#     articles = soup.find_all("div", class_="cnn-search__result-thumbnail")
#     for article in articles:
#         title = article.find("span", class_="cnn-search__result-headline").text
#         summary = article.find("div", class_="cnn-search__result-body").text
#         results.append({"title": title, "summary": summary})

#     return results

# query = "climate change"
# results = get_cnn_search_results(query)
# for result in results:
#     print("Title:", result["title"])
#     print("Summary:", result["summary"])
#     print()

# def freeze_support():
#  '''
#  Check whether this is a fake forked process in a frozen executable.
#  If so then run code specified by commandline and exit.
#  '''
#  if sys.platform == 'win32' and getattr(sys, 'frozen', False):
#      from multiprocessing.forking import freeze_support
#      freeze_support()

import newspaper
if __name__ == '__main__':
#   freeze_support()
  for x in range(300):
    url = "https://edition.cnn.com/search?q=climate+change&size=10&page="+str(x+1)+"&sort=newest&types=article&from=0&section="
    cnn_paper = newspaper.build(url, memoize_articles=False)  # ~15 seconds
    print(len(cnn_paper.articles))
    url_list = []
    for article in cnn_paper.articles:
        if article.url not in url_list:
            url_list.append(article.url)
    print(url_list)