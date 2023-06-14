import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from textblob import TextBlob
import spacy
from spacy.lang.en import English

nlp = spacy.load("en_core_web_sm")
def pos_tag(text):
    try:
        return TextBlob(text).tags
    except:
        return None
def get_adjectives(text):    
    blob = TextBlob(text)
    return [ word for (word,tag) in blob.tags if (tag == "NN" or tag == "NNS" or tag == "NNP" or tag == "NNPS" or tag == "RB" or tag == "RBR" or tag == "RBS" or tag == "RP" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ" or tag == "FW" )]
import matplotlib.pyplot as plt

from socket import SocketIO
import pandas as pd
import numpy as np
# Topic model
from bertopic import BERTopic
# Dimension reduction
from umap import UMAP

review_datasets = pd.read_excel(r'C:\Users\tretiak\Desktop\Topic_modeling\Articles_with_topics_6.0.xlsx')
reviews_datasets1 = pd.DataFrame(review_datasets, columns=['Articles'])
reviews_datasets2=reviews_datasets1.iloc [0:8000]
reviews_datasets4 = pd.DataFrame(review_datasets, columns=['Articles'])
reviews_datasets=reviews_datasets2[~reviews_datasets1.Articles.str.contains('None')]
reviews_datasets.head()
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import string
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora
import gensim.corpora as corpora
from nltk.stem import WordNetLemmatizer
from nltk.stem import porter
stop_words=set(nltk.corpus.stopwords.words('english'))
new_stopwords = ["Background", "Main body", "Conclusion", "Methodology", "Methods", "Results", "Conclusions", "also", "Aim","Materials", "Aims", "Objective", "Design", "Interpretation", "Objectives", "Context"]
new_stopwords.extend(['study',"bla",'activity','found','two','different','identified','expression','including','clinical','showed','patient','associated','one','result','using','may','increased','presence','used','effect','among','development','level','new','analysis','observed','target','novel','However','potential','concentration','system','role','specie','method','compound','important','respectively','well','increase','due','three','high','detected','show','combination','revealed','change','rate','involved','type','common','model','addition','several','growth','data','present','within','multiple','major','class','known','tested','demonstrated','group','compared','region','factor','complex','response','determined','production','effective','strategy','control','investigated','approach','specific','reduced','various','function','many','responsible', 'ability','four','significant','similar','related','variant','profile','case','total','pathway','sensitive','studied','report','selection','performed','suggesting','higher','understanding','shown','first','substitution','active','via','selected','increasing','caused','reported','number','test','element','finding','time','highly' ])

#print(new_stopwords)
exclude = set(string.punctuation)

def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count = bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod
def alphabet(ch):
    return ch.replace("α","alpha").replace("MICs","MIC").replace("β","beta").replace("γ","gamma").replace("δ","delta").replace("ε","epsilon").replace("ζ","zeta").replace("η","eta").replace("θ","theta").replace("ι","iota").replace("κ","kappa").replace("λ","lambda").replace("μ","mu").replace("ν","nu").replace("ξ","xi").replace("ο","omicron").replace("π","pi").replace("ρ","rho").replace("σ","sigma").replace("τ","tau").replace("υ","upsilon").replace("φ","phi").replace("χ","chi").replace("ψ","psi").replace("ω","omega")


def clean_text(headline):
    doc = nlp(headline)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_digit and len(token) > 1]
    cleaned_text1 = " ".join(alphabet(ch) for ch in lemmatized_tokens if ch not in exclude)
    cleaned_text2 = " ".join(ch for ch in cleaned_text1.split("."))
    #new_stopwords1 = " ".join(i.lemma_ for i in nlp(new_stopwords)) 
    cleaned_text=" ".join(ch for ch in cleaned_text2.split() if ch not in new_stopwords)
    return cleaned_text
reviews_datasets['cleaned_text']=reviews_datasets['Articles'].apply(clean_text)
reviews_datasets['words'] = reviews_datasets['cleaned_text'].apply(get_adjectives)

review_lemmatized = reviews_datasets['cleaned_text'] .tolist()
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# use tfidf by removing tokens that don't appear in at least 50 documents
vect = TfidfVectorizer(min_df=50, stop_words='english')
 
# Fit and transform
X = vect.fit_transform(review_lemmatized)

# Create an NMF instance: model
# the 10 components will be the topics
model = NMF(n_components=5, random_state=0)
 
# Fit the model to TF-IDF
model.fit(X)
 
# Transform the TF-IDF: nmf_features
nmf_features = model.transform(X)

def get_topic_nmf(doc):  
# New Document to predict
#new_doc = "Recurrent oral candidosis is a common problem in immunocompromised patients, and it is frequently triggered by resistance induced by antifungal treatment. Knowledge of the mechanisms by which the yeast persists in the host could allow the management of this type of infection. This study used electrophoretic karyotyping and restriction fragment length polymorphism based on the use of 27A probe to study 12 pairs of Candida albicans isolates from patients with recurrent candidosis to distinguish new infections from relapses caused by the same strain responsible for the first episode. Subsequently, RT-PCR was used to evaluate expression of CDR1, CDR2 and MDR1 genes, which are involved in C. albicans azole resistance, in the three pairs that consisted of variants of the same strain. Restriction polymorphism resulted in better discrimination than with karyotyping in defining differences between strains. In one case, RT-PCR allowed us to identify deregulation of efflux pump genes as the possible underlying mechanism in recurrent candidosis. The techniques employed resulted effective for the characterization of recurrent oral candidosis. Broader analysis could help to control better these infections and choose adequate therapy."

# Transform the TF-IDF
    X_new = vect.transform([doc])

# Transform the TF-IDF: nmf_features
    nmf_features_new = model.fit_transform(X_new)

# The idxmax function returns the index of the maximum element
# in the specified axis
    
    return format(pd.DataFrame(nmf_features_new).idxmax(axis=1).iloc[0] + 1)

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('try1.html', topic=None)

@app.route('/try1', methods=['POST', 'GET'])
def get_topic():
    topic = None
    if request.method == 'POST':
        user = request.form['nm']
        topic = get_topic_nmf(user) # Replace with your code to get the topic
    return render_template('try1.html', topic=topic)

if __name__ == '__main__':
    app.run(debug=True)