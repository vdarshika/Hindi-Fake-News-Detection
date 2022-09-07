#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from inltk.inltk import setup
from inltk.inltk import get_embedding_vectors
from inltk.inltk import remove_foreign_languages
from inltk.inltk import get_embedding_vectors
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import re
import string
from ast import arg
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# model
import os
import json
from ast import arg
import math
import torch.nn.functional as Fu
from numpy.linalg import norm
from torch.autograd import Variable
import sys
from tqdm import tqdm

# In[ ]:


setup('hi')
print('Hindi setup complete!')


# In[ ]:


batch_size = 11
rand_word_arr = np.random.random((2, 400))
rand_word_arr = rand_word_arr[:-1, ]

# preprocess data


def preprocessing(txt):
    output = remove_foreign_languages(txt, 'hi')
    clean = []
    for txt in output:
        txt = "".join([c for c in txt if c not in string.punctuation+'▁'])
        if not re.match(r'[A-Z]+', txt, re.I) and not txt == '':
            clean.append(txt)
    cleaned_headline = " ".join(clean)
    return cleaned_headline

# generate word_embedding


def get_encoding(context):
    encoding = []
    for word in context:
        encoding.append(np.array(get_embedding_vectors(word, 'hi')))
    return encoding

# convert every encoding into equal size which is size of max encoding


def pad_encoding(encoding, max_len):
    new_encoding = []
    for context in encoding:
        if len(context) < max_len:
            diff = max_len - len(context)
            for _ in range(diff):
                context = np.concatenate((context, rand_word_arr))
        new_encoding.append(context)

    new_encoding = np.array(new_encoding)
    return new_encoding

# convert data into list of dictionary


def get_data_list(new_headline_encoding, new_body_encoding, label):
    all_list_of_data = []
    for body, head, lbl in zip(new_body_encoding, new_headline_encoding, label):
        pres_dict = {}
        pres_dict["body"] = np.array(body)
        pres_dict["heading"] = np.array(head)
        pres_dict["label"] = lbl
        all_list_of_data.append(pres_dict)

    return all_list_of_data

# run batch of data samples


def run_batch(index, df_size):
    data = df[index * batch_size: min((index+1)*batch_size, df_size)]
    train_headline = data['heading'].apply(lambda x: preprocessing(x)).tolist()
    train_body = data['body'].apply(lambda x: preprocessing(x)).tolist()
    train_label = data['label'].tolist()
    train_headline_encoding = get_encoding(train_headline)
    train_body_encoding = get_encoding(train_body)
    del train_headline, train_body
    max_body_len = max(map(len, train_body_encoding))
    max_headline_len = max(map(len, train_headline_encoding))
    train_new_headline_encoding = pad_encoding(
        train_headline_encoding, max_headline_len)
    train_new_body_encoding = pad_encoding(train_body_encoding, max_body_len)
    del train_headline_encoding, train_body_encoding
    train_all_list_of_data = get_data_list(
        train_new_headline_encoding, train_new_body_encoding, train_label)
    del train_new_headline_encoding, train_new_body_encoding

    return (train_all_list_of_data, train_label, max_body_len, max_headline_len)


# In[ ]:

train_dataset = input("Enter the name of the train dataset : ")
train_path = "final-datasets/"+train_dataset
df = pd.read_csv(train_path)
df = pd.DataFrame(df, columns=['body', 'heading', 'label'])


# In[ ]:


class args:
    d = 400                 # Dimension of each word vector
    hidden_lstm_dim = 100   # Dimension of hidden layer
    ff_input_dim = 400      # No of nodes in input layer of FF model
    ff_hidden_dim = 100     # No of nodes in hidden layer of FF model
    ff_output_dim = 2       # No of nodes in output layer of FF model


# In[ ]:


class HindiModel(nn.Module):
    def __init__(self):
        super(HindiModel, self).__init__()
        self.LSTM_head = nn.LSTM(num_layers=1, input_size=args.d,
                                 hidden_size=int(args.hidden_lstm_dim),
                                 batch_first=True)
        self.LSTM_body = nn.LSTM(num_layers=1, input_size=args.d,
                                 hidden_size=int(args.hidden_lstm_dim),
                                 batch_first=True)
        self.feed_forward = nn.Sequential(nn.Linear(args.ff_input_dim, args.ff_hidden_dim),
                                          nn.Sigmoid(),
                                          nn.Linear(args.ff_hidden_dim,
                                                    args.ff_output_dim),
                                          nn.Sigmoid())

    def get_lstm_encoding(self, all_news_heading_body, max_body_len, max_headline_len):
        all_lstm_hidden_state_head = []
        all_lstm_hidden_state_body = []
        for pres_head_body in all_news_heading_body:
            pres_heading = pres_head_body['heading']
            pres_body = pres_head_body['body']
            head_tensor = torch.tensor(pres_heading, dtype=torch.double).view(
                1, max_headline_len, args.d)
            body_tensor = torch.tensor(pres_body, dtype=torch.double).view(
                1, max_body_len, args.d)
            encoded_body, (hidden_out_body, _) = self.LSTM_body(body_tensor)
            encoded_head, (hidden_out_head, _) = self.LSTM_head(head_tensor)
            all_lstm_hidden_state_head.append(
                hidden_out_head.view(1, args.hidden_lstm_dim))
            all_lstm_hidden_state_body.append(
                hidden_out_body.view(1, args.hidden_lstm_dim))

        return all_lstm_hidden_state_head, all_lstm_hidden_state_body

    def forward(self, data, max_body_len, max_headline_len):
        all_lstm_hidden_state_head, all_lstm_hidden_state_body = self.get_lstm_encoding(
            data, max_body_len, max_headline_len)
        outputs = []
        for X, Y in zip(all_lstm_hidden_state_head, all_lstm_hidden_state_body):
            XminusY = X - Y
            XdotY = X * Y
            feed_forward_input_vector = torch.cat(
                [X, XdotY, XminusY, Y], dim=1)
            feed_forward_output = self.feed_forward(feed_forward_input_vector)
            outputs.append(feed_forward_output[0])
        outputs = torch.stack(outputs)

        return outputs


# In[ ]:

model = HindiModel().double()
criterion = torch.nn.CrossEntropyLoss()
model.train()
params = [p for p in model.parameters() if p.requires_grad]
print("plen:", len(params))

optimizer = torch.optim.SGD([{'params': params}], lr=0.1)


# In[ ]:


for epoch in range(100):
    no_of_samples = len(df)
    no_of_batches = int(no_of_samples/batch_size)

    no_of_iterations = 0
    if no_of_samples % batch_size == 0.0:
        no_of_iterations = no_of_batches
    else:
        no_of_iterations = no_of_batches + 1

    for index in tqdm(range(no_of_iterations)):
        all_list_of_data, train_label, max_body_len, max_headline_len = run_batch(
            index, no_of_samples)
        y = torch.from_numpy(np.array(train_label)).long()
        y_cap = model.forward(all_list_of_data, max_body_len, max_headline_len)
        loss = criterion(y_cap, y)
        loss.requres_grad = True
        loss.retain_grad()

        # Zero gradients, perform a backward pass, and update the weights.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        del all_list_of_data
    torch.save(model.state_dict(), "model_"+str(epoch)+".pth")
    print('epoch {}, loss {}'.format(epoch, loss.item()))
