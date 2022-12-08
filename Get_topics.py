import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from textblob import TextBlob
def pos_tag(text):
    try:
        return TextBlob(text).tags
    except:
        return None
def get_adjectives(text):
    blob = TextBlob(text)
    return [ word for (word,tag) in blob.tags if (tag == "NN" or tag == "NNS" or tag == "NNP" or tag == "NNPS" or tag == "RB" or tag == "RBR" or tag == "RBS" or tag == "RP" or tag == "VB" or tag == "VBG" or tag == "VBP" or tag == "VBZ" or tag == "FW" )]
import matplotlib.pyplot as plt

from socket import SocketIO
import pandas as pd
import numpy as np

review_datasets = pd.read_excel(r'C:\Users\tretiak\Desktop\Topic_modeling\Articles_with_topics_6.0.xlsx')
reviews_datasets1 = pd.DataFrame(review_datasets, columns=['Articles'])
reviews_datasets2=reviews_datasets1.iloc [0:8000]
reviews_datasets=reviews_datasets2[~reviews_datasets1.Articles.str.contains('None')]
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
new_stopwords.extend(['study','population','environment','host','significantly','interaction','site','previously','stress','binding','transfer','prevalence','condition','identify','action','review','process','provide','year','play','spread','range','difference','efficacy','culture','Furthermore','formation','leading','problem','family','property','often','research','sample','containing','community','however','work','promoter','especially','impact','loss','Moreover','challenge','Thus','recently','medium','confer','commonly','thus','importance','diversity','absence','indicating','resulting','provides','evidence','basis','suggests','decrease','rapidly','remains','Therefore','even','inactivation','surface','mainly','relationship','knowledge','regulation','vivo','transport','experiment','particularly','setting','concern','Using','carrying','position','indicate',"bla",'activity','found','two','different','identified','expression','including','clinical','showed','patient','associated','one','result','using','may','increased','presence','used','effect','among','development','level','new','analysis','observed','target','novel','However','potential','concentration','system','role','specie','method','compound','important','respectively','well','increase','due','three','high','detected','show','combination','revealed','change','rate','involved','type','common','model','addition','several','growth','data','present','within','multiple','major','class','known','tested','demonstrated','group','compared','region','factor','complex','response','determined','production','effective','strategy','control','investigated','approach','specific','reduced','various','function','many','responsible', 'ability','four','significant','similar','related','variant','profile','case','total','pathway','sensitive','studied','report','selection','performed','suggesting','higher','understanding','shown','first','substitution','active','via','selected','increasing','caused','reported','number','test','element','finding','time','highly' ])

exclude = set(string.punctuation)

def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count = bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod
def alphabet(ch):
    return ch.replace("α","alpha").replace("MICs","MIC").replace("β","beta").replace("γ","gamma").replace("δ","delta").replace("ε","epsilon").replace("ζ","zeta").replace("η","eta").replace("θ","theta").replace("ι","iota").replace("κ","kappa").replace("λ","lambda").replace("μ","mu").replace("ν","nu").replace("ξ","xi").replace("ο","omicron").replace("π","pi").replace("ρ","rho").replace("σ","sigma").replace("τ","tau").replace("υ","upsilon").replace("φ","phi").replace("χ","chi").replace("ψ","psi").replace("ω","omega")


def clean_text(headline):
    le=WordNetLemmatizer()
    word_tokens=word_tokenize(headline)
    tokens=" ".join([le.lemmatize(i) for i in word_tokens if (( i.lower() not in stop_words) and ( not  i.isdigit()) and (len(i)>1) and ( le.lemmatize(i) not in stop_words) ) ])
    cleaned_text1=" ".join(alphabet(ch) for ch in tokens.split() if (ch not in exclude ))
    cleaned_text2=" ".join(ch for ch in cleaned_text1.split(".") )
    new_stopwords1=" ".join([le.lemmatize(i) for i in new_stopwords]) 
    stop_words1=" ".join([le.lemmatize(i) for i in stop_words])
    cleaned_text3=" ".join(ch for ch in cleaned_text2.split() if ( ( not  ch.isdigit()) and (len(ch)>1)))
    cleaned_text=" ".join(ch for ch in cleaned_text3.split() if ( ch not in new_stopwords1 ) and ( ch not in stop_words1 ) )
    return cleaned_text
reviews_datasets['cleaned_text']=reviews_datasets['Articles'].apply(clean_text)
reviews_datasets['words'] = reviews_datasets['cleaned_text'].apply(get_adjectives)
doc_clean=[]
for doc in reviews_datasets['cleaned_text']:
    doc_clean.append(get_adjectives(doc))
    
writer = pd.ExcelWriter('Adjectives5.xlsx')
reviews_datasets.to_excel(writer)
writer.save()

bigram_mod = bigrams(doc_clean)
bigram = [bigram_mod[review] for review in doc_clean ]
#bigram=[get_adjectives(words) for words in bigram1]
#creating LDA model
id2word = corpora.Dictionary(bigram)
print("Without filter extreme",len(id2word))
id2word.filter_extremes(no_below=1, no_above=0.4)
print("With filter extreme",len(id2word))
corpus = [id2word.doc2bow(text) for text in bigram]


#creating LDA model
Lda = gensim.models.ldamodel.LdaModel
from gensim.test.utils import datapath

ldamodel = Lda(corpus=corpus , id2word=id2word, passes=10, iterations=10, random_state=42, num_topics=5, decay=0.5, offset=1, alpha="auto", eta="auto")
ldamodel.save('lda_train.model')
#printing topics
for idx, topic in ldamodel.print_topics(-1,num_words=30 ):
    print("Topic: {} \nWords: {}".format(idx, topic))
    print("\n")

# get the model's topics in their native ordering...
all_topics = ldamodel.print_topics(num_words=30)
# create an empty list per topic to collect the docs:
docs_per_topic = [[] for _ in all_topics]
# create an empty list per document to collect the topics:
topics_per_document=[[] for i in corpus ]
# create an empty list for collecting dominant topics for documents:
topic_out=[]
weight_out=[]


zero=0
first=0
second=0
third=0
fourth=0
# for every doc...
for doc_id, doc_bow in enumerate(corpus ):
    # get its topics
    doc_topics = ldamodel.get_document_topics(doc_bow)
    # and  for each of its topics
    for topic_id, score in doc_topics:
        # add the topic_id & its score to the each document list
         topics_per_document[doc_id].append((topic_id, score))
         docs_per_topic[topic_id].append((doc_id, score))
for i in topics_per_document:
    #sort topics by their score
    main_topic=sorted(list(i),key=lambda x: x[1], reverse=True)
    #add the dominant one
    topic_out.append(main_topic[0][0])
    weight_out.append(main_topic[0][1])
    if(main_topic[0][0]==0 and main_topic[0][1]>=0.5):
        zero+=1
    if(main_topic[0][0]==1 and main_topic[0][1]>=0.5):
        first+=1
    if(main_topic[0][0]==2 and main_topic[0][1]>=0.5):
        second+=1
    if(main_topic[0][0]==3 and main_topic[0][1]>=0.5):
        third+=1 
    if(main_topic[0][0]==4 and main_topic[0][1]>=0.5):
       fourth+=1



reviews_datasets['Topic'] = topic_out
reviews_datasets['Topic weight'] = weight_out
reviews_datasets['colFromIndex'] = reviews_datasets.index
#reviews_datasets=reviews_datasets.dropna()
writer = pd.ExcelWriter('CompareTopics_withoutpastxlsx')
reviews_datasets.sort_values( by =['Topic','colFromIndex']).to_excel(writer)
writer.save()


words = [word for words in bigram for word in words]
#print(words)
fdist=nltk.FreqDist(words)
top200 = fdist.most_common(200)
for i in top200:
    print(i)


topic_zero = reviews_datasets[reviews_datasets['Topic'] == 0]
print('Number of documents with zero topic: ', len(topic_zero))
#topic_zero2=0
#for i in review_datasets:
#if ((review_datasets['Topic'] == 0) and (review_datasets['Topic weight'] >= 0.5)):
   # topic_zero2+=1 
topic_zero1=reviews_datasets['Topic weight'] .where(reviews_datasets['Topic'] == 0)
#topic_zero2=topic_zero1.where(review_datasets['Topic weight'] >= 0.5)
print('Number of documents with zero topic above 0.5: ', zero)
#print(topic_zero1)

topic_first = reviews_datasets[reviews_datasets['Topic'] == 1]
print('Number of documents with first topic: ', len(topic_first))
print('Number of documents with first topic above 0.5: ', first)

topic_second = reviews_datasets[reviews_datasets['Topic'] == 2]
print('Number of documents with second topic: ', len(topic_second))
print('Number of documents with second topic above 0.5: ', second)

topic_third = reviews_datasets[reviews_datasets['Topic'] == 3]
print('Number of documents with third topic: ', len(topic_third))
print('Number of documents with third topic above 0.5: ', third)

topic_fourth = reviews_datasets[reviews_datasets['Topic'] == 4]
print('Number of documents with fourth topic: ', len(topic_fourth))
print('Number of documents with fourth topic above 0.5: ', fourth)
# Compute Perplexity Score
print('nPerplexity Score: ', ldamodel.log_perplexity(corpus))

# Compute Coherence Score
coherence_model_lda = gensim.models.CoherenceModel(model=ldamodel, texts=bigram, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('nCoherence Score: ', coherence_lda)


print("Alpha score: ", ldamodel.alpha)
print("Eta score: ", ldamodel.eta)

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  
import matplotlib.pyplot as plt
%matplotlib inline

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus, id2word)
vis

from collections import Counter
import matplotlib.colors as mcolors
topics = ldamodel.show_topics(formatted=False)
data_flat = [w for w_list in bigram for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        
writer = pd.ExcelWriter('WordsCount.xlsx')
df.to_excel(writer)
writer.save()
# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 3, figsize=(20,14), sharey=True, dpi=160)
axes[1][2].set_visible(False)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.05); ax.set_ylim(0, 25000)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()