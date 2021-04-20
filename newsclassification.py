import pandas as pd 
import numpy as np

from sklearn.metrics import classification_report
import string, re

fake_news=pd.read_csv('Fake.csv') 
legit_news = pd.read_csv('True.csv')


#how data looks
fake_news.head(10)
legit_news.head(10)


#Classify news as 0/1
fake_news['Class']=0 
legit_news['Class']=1


#merging fake and true news datasets

df1 = pd.concat([fake_news, legit_news], axis=0)


#dropping the unneccesary coloums 
new_data = df1.drop(["title", "subject", "date"], axis=1)

#shuffling the dataset
new_data= new_data.sample(frac=1)

#Drop All the special Characters, spaces from the dataset
def char_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text
new_data["text"] = new_data["text"].apply(char_drop)

#SEPARATIN THE DEPENDENT AND INDEPENDENT VARIABLE
X = new_data['text']
y = new_data['Class']

#TRAINN TEST DATA

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)



#Feautre Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
Xv_train = vectorization.fit_transform(X_train)
Xv_test = vectorization.transform(X_test)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
Classification = LogisticRegression()
Classification.fit(Xv_train, y_train)


Classification.score(Xv_test,y_test)
Classification.score(Xv_train, y_train)


y_pred = Classification.predict(Xv_test)

#How many corrections are model got right
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cls_rep = classification_report(y_test, y_pred)

#DECISION TREE CLASSIFIER

from sklearn.tree import DecisionTreeClassifier
dt_clf= DecisionTreeClassifier()
dt_clf.fit(Xv_train, y_train)


dt_clf.score(Xv_test,y_test)
dt_clf.score(Xv_train, y_train)


y_pred = Classification.predict(Xv_test)

dcm = confusion_matrix(y_test, y_pred)

dc_cls = classification_report(y_test, y_pred)


















