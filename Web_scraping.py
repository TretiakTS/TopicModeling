from doctest import Example
from sre_parse import State
import urllib.request,sys,time
from bs4 import BeautifulSoup
import requests
import pandas as pd

pagesToGet= 1422  
upperframe=[]  
for page in range(1,pagesToGet+1):
    print('processing page :', page)
    url = 'https://pubmed.ncbi.nlm.nih.gov/?term=Mechanisms+of+Antibiotic+Resistance&filter=simsearch2.ffrft&page='+str(page)
    try:
        
        page=requests.get(url)                             
    except Exception as e:                                   
        error_type, error_obj, error_info = sys.exc_info()      
        print ('ERROR FOR LINK:',url)                          
        print (error_type, 'Line:', error_info.tb_lineno)     
        continue                                             
    #time.sleep(1)   
    soup=BeautifulSoup(page.text,'html.parser')
    frame=[]
    links=soup.find_all('article',attrs={'class':'full-docsum'})
    for j in links:
        Links = "https://pubmed.ncbi.nlm.nih.gov/"
        Links += j.find("div",attrs={'class':'docsum-content'}).find('a')['href'].strip()
        try:
            page2=requests.get(Links)
        except Exception as e:                                  
            error_type, error_obj, error_info = sys.exc_info()     
            print ('ERROR FOR LINK:',Links)                         
            print (error_type, 'Line:', error_info.tb_lineno)     
            continue                                            
        #time.sleep(1)
        soup2=BeautifulSoup(page2.text,'html.parser')
        try:
            Articles=""
            for Statements in soup2.find("div",attrs={'class':'abstract-content selected'}).find_all('p'):
                Articles+=Statements.text.strip()
        except Exception as e:
            Articles='None'
        frame.append((Articles))
    upperframe.extend(frame)
import numpy as np
reviews_datasets = pd.DataFrame(upperframe, columns=['Articles'])
from sklearn.feature_extraction.text import CountVectorizer

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import string
stop_words=set(nltk.corpus.stopwords.words('english'))
exclude = set(string.punctuation)
def clean_text(headline):
    le=WordNetLemmatizer()
    word_tokens=word_tokenize(headline)
    tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words]
    cleaned_text1=" ".join(tokens)
    cleaned_text=cleaned_text1.translate(str.maketrans('', '', string.punctuation))
    return cleaned_text.lower()
reviews_datasets['cleaned_text']=reviews_datasets['Articles'].apply(clean_text)
vect =TfidfVectorizer(stop_words=stop_words, max_df=0.9, min_df=10)
vect_text=vect.fit_transform(reviews_datasets['cleaned_text'])
from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=5,random_state=42,max_iter=500)
LDA.fit(vect_text)
vocab = vect.get_feature_names_out()
for i, comp in enumerate(LDA.components_):
     vocab_comp = zip(vocab, comp)
     sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
     print("Topic "+str(i)+": ")
     for t in sorted_words:
            print(t[0],end=" ")
     print('\n')
topic_values = LDA.transform(vect_text)
topic_values.shape
reviews_datasets['Topic'] = topic_values.argmax(axis=1)
reviews_datasets.head(n=14215)

writer = pd.ExcelWriter('Articles_with_topics_6.0.xlsx')
reviews_datasets.to_excel(writer)
writer.save()