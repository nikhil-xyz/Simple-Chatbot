import json
import pandas as pd
import numpy as np
import string
import random
import pickle
import streamlit as st
from streamlit_chat import message

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from datetime import date, datetime
date = date.today()
time = datetime.now()

f = open('content.json')
temp = json.load(f)

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
responses = pickle.load(open('responses.pkl', 'rb'))

model = load_model('model.h5')


# Storing the chat
if 'maintained' not in st.session_state:
    st.session_state['maintained'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def generate():
  user_input = st.session_state.input_text
  st.session_state.input_text = ""

  texts_p = []
  prediction_input = [letters.lower() for letters in user_input if letters not in string.punctuation]
  prediction_input = ''.join(prediction_input)

  texts_p.append(prediction_input)

  prediction_input = tokenizer.texts_to_sequences(texts_p)
  prediction_input = np.array(prediction_input).reshape(-1)
  prediction_input = pad_sequences([prediction_input], 7)

  output = model.predict(prediction_input)
  index = output.argmax()

  print(output)

  response_tag = encoder[index]
  message_bot = random.choice(responses[response_tag])
  print(response_tag)

  if response_tag == "datetime":
    message_bot = str(time)[:-10]


  st.session_state.past.append(user_input)
  st.session_state.maintained.append(message_bot)

  for i in range(len(st.session_state['maintained'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    message(st.session_state["maintained"][i], key=str(i))


st.text_input("say something", key='input_text', on_change=generate)









