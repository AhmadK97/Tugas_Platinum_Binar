import streamlit as st
import numpy as np
import pickle
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
import torch
import torch.nn.functional as F
from data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model as klm
from tensorflow.keras.preprocessing.text import Tokenizer
from jcopml.utils import load_model
import pandas as pd

#Model Bert

tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
config = BertConfig.from_pretrained(r"C:\Users\ahmad\Project\Tugas Akhir\PretrainedBART\config.json")
config.num_labels = DocumentSentimentDataset.NUM_LABELS

modelBERT = BertForSequenceClassification.from_pretrained(r"C:\Users\ahmad\Project\Tugas Akhir\PretrainedBART\pytorch_model.bin", 
                                                      config=config)

w2i, i2w = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL
st.title('Sentiment Predictor!')

#Model LSTM

modelLSTM = klm('LSTM.h5')

#Model NNSklearnModel

modelMLPC = load_model('MLPCPlatinumTfidf0.88%.pkl')


#Kalkulasi Model Bert

box2 = st.selectbox('Select the Model', ['NNSklearnModel','LSTMModel','Bert'])
box1 = st.text_input('Input Text')


subwords = tokenizer.encode(box1)
subwords = torch.LongTensor(subwords).view(1, -1).to(modelBERT.device)

logits = modelBERT(subwords)[0]
label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
hasilBERT = f'{i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)'

#Kalkulasi Model LSTM

def cleaningpertama(text):
    re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = re.sub('  +', ' ', text)
    return text

with open('tokenizer.pickle', 'rb') as handle: tokenizer = pickle.load(handle)

sentiment = ['Negative', 'Neutral', 'Positive']

text = [cleaningpertama(box1)]
predicted = tokenizer.texts_to_sequences(text)
guess = pad_sequences(predicted, maxlen=85)

prediction = modelLSTM.predict(guess)
polarity = np.argmax(prediction[0])
hasilLSTM = f'{sentiment[polarity]}'


#Kalkulasi Model NNSklearnModel

MLPCpredict = (modelMLPC.predict(pd.DataFrame([cleaningpertama(box1)])[0]))[0]

#---------------------------------------------------------------------------------#

if st.button('Submit!'):
    if box2 == 'Bert':
        st.title(hasilBERT)
    if box2 == 'LSTMModel':
        st.title(hasilLSTM)
    if box2 == 'NNSklearnModel':
        st.title(MLPCpredict)

if st.button('to file JSON'):
    if box2 == 'Bert':
        st.title(hasilBERT.to_json())
    if box2 == 'LSTMModel':
        st.title(hasilLSTM.to_json())
    if box2 == 'NNSklearnModel':
        st.title(MLPCpredict.to_json())