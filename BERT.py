#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from inltk.inltk import remove_foreign_languages 
from inltk.inltk import setup
import string
# setup('en')
# setup('hi')


# In[42]:


train_set=input("Enter the name of the train dataset : ")
test_set=input("Enter the name of the test dataset : ")
train_path="final-datasets/"+train_set;
test_path="final-datasets/"+test_set;
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print(type(train))


# In[6]:


print(train.head(5))
print(test.head(5))
test


# In[7]:


train=train.dropna()
train = train.reset_index(drop = True)

test=test.dropna()
test = test.reset_index(drop = True)


# In[8]:


def clean_txt(txt):
    output = remove_foreign_languages(txt, 'hi')
    # output
    clean = []
    for txt in output:
        txt = "".join([c for c in txt if c not in string.punctuation+'▁'])
        if not re.match(r'[A-Z]+', txt, re.I) and not txt == '':
            clean.append(txt)
    cleaned_headline = " ".join(clean)
    return cleaned_headline


# In[9]:


setup('hi')


# In[10]:


#data cleaning for train data
headline_train = train['heading'].apply(lambda x:clean_txt(x)).tolist()
body_train = train['body'].apply(lambda x:clean_txt(x)).tolist()
label_train = train['label'].tolist()


# In[11]:


#data cleaning for test data
headline_test = test['heading'].apply(lambda x:clean_txt(x)).tolist()
body_test = test['body'].apply(lambda x:clean_txt(x)).tolist()
label_test = test['label'].tolist()


# In[34]:


headline_train = pd.Series(headline_train)
body_train = pd.Series(body_train)
label_train = pd.Series(label_train)


# In[39]:


temp = {"heading": headline_train,
        "body": body_train,
        "label": label_train}


# In[40]:


clean_data = pd.concat(temp,
               axis = 1)


# In[43]:


train_path=""
print(train_set)
print(test_set)
train_path="final-datasets/cleaned_"+train_set;
clean_data.to_csv(train_path)


# In[ ]:





# In[44]:


headline_test = pd.Series(headline_test)
body_test = pd.Series(body_test)
label_test = pd.Series(label_test)


# In[45]:


temp = {"heading": headline_test,
        "body": body_test,
        "label": label_test}


# In[46]:


clean_data = pd.concat(temp,
               axis = 1)


# In[47]:


test_path=""
test_path="final-datasets/cleaned_"+test_set;
clean_data.to_csv(test_path)


# In[ ]:





# In[48]:


clean_data_train=pd.read_csv(train_path)
clean_data_test=pd.read_csv(test_path)


# In[49]:


clean_data_train=clean_data_train.dropna()
clean_data_train = clean_data_train.reset_index(drop = True)
clean_data_test=clean_data_test.dropna()
clean_data_test = clean_data_test.reset_index(drop = True)


# In[50]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install SentencePiece')


# In[51]:


temp_body_test=clean_data_test['body']
temp_heading_test=clean_data_test['heading']
temp_label_test=clean_data_test['label'].to_numpy()
temp_label_test


# In[52]:


temp_body_test


# In[25]:


print(len(temp_label_test))


# In[26]:


temp_body_train=clean_data_train['body']
temp_heading_train=clean_data_train['heading']
temp_label_train=clean_data_train['label'].to_numpy()
temp_label_train


# In[27]:


temp_body_train


# In[28]:


print(len(temp_label_train))


# In[ ]:





# In[29]:


# from transformers import BertTokenizer
# from transformers import TFBertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
# model = TFBertModel.from_pretrained('bert-base-multilingual-uncased')
# ei=tokenizer(temp[0],return_tensors='tf',max_length = 512,pad_to_max_length = True)
# o=model(ei)
# Load the BERT tokenizer. #bert-base-uncased


# In[30]:


# sentence_vector=o[0].numpy()[0][3]
# print(len(sentence_vector))
# print(type(sentence_vector))
# print(sentence_vector)


# In[31]:


from transformers import BertTokenizer
from transformers import TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = TFBertModel.from_pretrained('bert-base-multilingual-uncased')


# In[32]:


def vectorize(heading,body):
    encoded_heading=tokenizer(heading,return_tensors='tf',max_length = 512,pad_to_max_length = True)
    encoded_body=tokenizer(body,return_tensors='tf',max_length = 512,pad_to_max_length = True)
    o_h=model(encoded_heading)
    o_b=model(encoded_body)
    vector_h=o_h[0].numpy()[0][3]
    vector_b=o_b[0].numpy()[0][3]
    vector_x=vector_h
    vector_y=vector_h
    for i in range(768):
        vector_x[i]=(vector_b[i]-vector_h[i])
        vector_y[i]=(vector_b[i]*vector_h[i])
    return np.concatenate((vector_h,vector_x,vector_y,vector_b), axis=0)


# In[33]:


vectors_train=[]
for i in range(len(temp_heading_train)):
    print(i)
    temp_vec=vectorize(temp_heading_train[i],temp_body_train[i])
    vectors_train.append(temp_vec);


# In[ ]:


vectors_train=np.asarray(vectors_train)


# In[ ]:


print(len(vectors_train[23]))


# In[ ]:


vectors_test=[]
for i in range(len(temp_heading_test)):
    print(i)
    temp_vec=vectorize(temp_heading_test[i],temp_body_test[i])
    vectors_test.append(temp_vec);


# In[ ]:


vectors_test=np.asarray(vectors_test)


# In[ ]:


print(len(vectors_test[23]))


# In[ ]:


print(vectors_train.shape)
print(vectors_test.shape)


# In[ ]:


vectors=np.concatenate((vectors_train, vectors_test), axis=0)
vectors.shape


# In[ ]:


temp_label=np.concatenate((temp_label_train, temp_label_test), axis=0)
temp_label.shape


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(vectors, temp_label, test_size=0.3,random_state=123, shuffle=True)
print((x_train))


# In[ ]:


# y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
# y_test = np.asarray(y_test).astype('float32').reshape((-1,1))


# In[ ]:


fcnn_model=Sequential()
fcnn_model.add(Dense(3072,input_dim=3072))
fcnn_model.add(Dense(500))
fcnn_model.add(Dense(1, activation='softmax'))


# In[ ]:


fcnn_model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


fcnn_model.fit(x_train,y_train,validation_split=0.1,epochs=100)


# In[ ]:


y_pred=fcnn_model.predict(x_test)


# In[ ]:


print("FCNN")
print("--------------------------------------------------------------")
print("Accuracy:\n",metrics.accuracy_score(y_test, y_pred))
print("\nperformance according to accuracy:\n",metrics.classification_report(y_test, y_pred))
print("--------------------------------------------------------------\n\n")


# In[ ]:





# In[ ]:


from sklearn.tree import DecisionTreeClassifier
import time


# In[ ]:


DT_clf = DecisionTreeClassifier()
start_time = time.time()
DT_clf = DT_clf.fit(x_train,y_train)
print("--- %s seconds to train the model---" % (time.time() - start_time))


# In[ ]:


y_pred_dt = DT_clf.predict(x_test)


# In[ ]:


print("decision tree")
print("--------------------------------------------------------------")
print("Accuracy:\n",metrics.accuracy_score(y_test, y_pred_dt))
print("\nperformance according to accuracy:\n",metrics.classification_report(y_test, y_pred_dt))
print("--------------------------------------------------------------\n\n")


# In[ ]:


from sklearn.svm import SVC
SVM_clf = SVC(kernel='poly')


# In[ ]:





# In[ ]:


start_time = time.time()
SVM_clf.fit(x_train, y_train)
print("--- %s seconds to train the model---" % (time.time() - start_time))


# In[ ]:


y_pred_svm = SVM_clf.predict(x_test)
print("SVM")
print("--------------------------------------------------------------")
print("Accuracy:\n",metrics.accuracy_score(y_test, y_pred_svm))
print("\nperformance according to accuracy:\n",metrics.classification_report(y_test, y_pred_svm))
print("--------------------------------------------------------------\n\n")


# In[ ]:


cm = confusion_matrix(y_test, y_pred_svm)
print(cm)


# In[ ]:


print("now using the indic bert\n")


# In[ ]:





# In[ ]:


clean_data_train=pd.read_csv(train_path)
clean_data_test=pd.read_csv(test_path)


# In[ ]:


clean_data_train=clean_data_train.dropna()
clean_data_train = clean_data_train.reset_index(drop = True)
clean_data_test=clean_data_test.dropna()
clean_data_test = clean_data_test.reset_index(drop = True)


# In[ ]:


temp_body_train=clean_data_train['body']
temp_heading_train=clean_data_train['heading']
temp_label_train=clean_data_train['label'].to_numpy()
temp_label_train


# In[ ]:


temp_body_test=clean_data_test['body']
temp_heading_test=clean_data_test['heading']
temp_label_test=clean_data_test['label'].to_numpy()
temp_label_test


# In[ ]:


from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
model = AutoModel.from_pretrained('ai4bharat/indic-bert')


# In[ ]:


def vectorize_in(heading,body):
    encoded_heading=tokenizer(heading,max_length = 512,pad_to_max_length = True,return_tensors="pt")["input_ids"]
    encoded_body=tokenizer(body,max_length = 512,pad_to_max_length = True,return_tensors="pt")["input_ids"]
    o_h=model(encoded_heading)
    o_b=model(encoded_body)
    vector_h=o_h[0][0][3]
    vector_b=o_b[0][0][3]
    vector_x=vector_h
    vector_y=vector_h
    for i in range(768):
        vector_x[i]=(vector_b[i]-vector_h[i])
        vector_y[i]=(vector_b[i]*vector_h[i])
    vector_h=vector_h.detach().numpy()
    vector_b=vector_b.detach().numpy()
    vector_x=vector_x.detach().numpy()
    vector_y=vector_y.detach().numpy()
    return np.concatenate((vector_h,vector_x,vector_y,vector_b), axis=0)


# In[ ]:


vectors_train=[]
for i in range(len(temp_heading_train)):
    print(i)
    temp_vec=vectorize_in(temp_heading_train[i],temp_body_train[i])
    vectors_train.append(temp_vec);


# In[ ]:


vectors_test=[]
for i in range(len(temp_heading_test)):
    print(i)
    temp_vec=vectorize_in(temp_heading_test[i],temp_body_test[i])
    vectors_test.append(temp_vec);


# In[ ]:


vectors_train=np.asarray(vectors_train)
vectors_test=np.asarray(vectors_test)


# In[ ]:


print(vectors_train.shape)
print(vectors_test.shape)

vectors=np.concatenate((vectors_train, vectors_test), axis=0)
print(vectors.shape)

temp_label=np.concatenate((temp_label_train, temp_label_test), axis=0)
print(temp_label.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(vectors, temp_label, test_size=0.3,random_state=123, shuffle=True)
print((x_train))


# In[ ]:


fcnn_model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


fcnn_model.fit(x_train,y_train,validation_split=0.1,epochs=100)


# In[ ]:


y_pred=fcnn_model.predict(x_test)


# In[ ]:


print("FCNN")
print("--------------------------------------------------------------")
print("Accuracy:\n",metrics.accuracy_score(y_test, y_pred))
print("\nperformance according to accuracy:\n",metrics.classification_report(y_test, y_pred))
print("--------------------------------------------------------------\n\n")


# In[ ]:





# In[ ]:





# In[ ]:


DT_clf = DecisionTreeClassifier()
start_time = time.time()
DT_clf = DT_clf.fit(x_train,y_train)
print("--- %s seconds to train the model---" % (time.time() - start_time))


# In[ ]:


y_pred_dt = DT_clf.predict(x_test)


# In[ ]:


print("decision tree")
print("--------------------------------------------------------------")
print("Accuracy:\n",metrics.accuracy_score(y_test, y_pred_dt))
print("\nperformance according to accuracy:\n",metrics.classification_report(y_test, y_pred_dt))
print("--------------------------------------------------------------\n\n")


# In[ ]:





# In[ ]:





# In[ ]:


start_time = time.time()
SVM_clf.fit(x_train, y_train)
print("--- %s seconds to train the model---" % (time.time() - start_time))


# In[ ]:


y_pred_svm = SVM_clf.predict(x_test)
print("SVM")
print("--------------------------------------------------------------")
print("Accuracy:\n",metrics.accuracy_score(y_test, y_pred_svm))
print("\nperformance according to accuracy:\n",metrics.classification_report(y_test, y_pred_svm))
print("--------------------------------------------------------------\n\n")


# In[ ]:




