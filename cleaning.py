# author yanfang.guo@vub.be Dec.16th 2017
import nltk

from csv_helper import *

tweets =  CSVHelper.load_csv("./Dataset/Tweets_2016London.csv")

# print out some examples
tweets_token = []
for i in range(3):
    print(tweets[i])

# 1.Tokenization
tknzr = nltk.TweetTokenizer(strip_handles=True, reduce_len=True)
for i in  range(len(tweets)):
    tweets_token.append(tknzr.tokenize(tweets[i]))

# print out some examples
for i in range(3):
    print(tweets_token[i])


# 2. Stop-word removal
#download the nltk stopwords
#nltk.download('stopwords')
stop_words_eng = nltk.corpus.stopwords.words('english')
tweets_del_stword = []
for i in tweets_token:
    temp = [w for w in i if w.lower() not in stop_words_eng]
    tweets_del_stword.append(temp)

'''
# testfunction to see if it works

length = []
for i in range(len(tweets)):
    length.append([len(tweets_token[i]),len(tweets_del_stword[i])])

print(length)
'''

# 3. stemming


tweets_stem = []
eng_stemmer = nltk.stem.snowball.EnglishStemmer(ignore_stopwords=False)
for i in tweets_del_stword:
    temp=[]
    for j in range(len(i)):
        temp.append(eng_stemmer.stem(i[j]))
    tweets_stem.append(temp)

# it seems that there exists some problems in this algorithms, such as eng_stemmer.stem(every) -> everi
for i in range(5):
    print(tweets_stem[i])
    print(tweets_del_stword[i])






