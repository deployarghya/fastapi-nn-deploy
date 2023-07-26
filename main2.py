import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app = FastAPI()

class TranscriptRequest(BaseModel):
    transcript: str

classes = ['depression', np.nan, 'ptsd', 'aspergers', 'ocd', 'adhd']

# Load the model and tokenizer
model_path = 'medicograd_V2_model.h5'
model = load_model(model_path)

tokenizer_path = 'tokenizer.json'
with open(tokenizer_path, 'r') as tokenizer_file:
    word_index = json.load(tokenizer_file)
    tokenizer = Tokenizer(num_words=100000)
    tokenizer.word_index = word_index

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


@app.post("/predict/")
def predict_topic(transcript_request: TranscriptRequest):
    user_input = transcript_request.transcript
    sequence = tokenizer.texts_to_sequences([preprocess_text(user_input)])
    sequence = pad_sequences(sequence, maxlen=100)

    prediction = model.predict(sequence)
    predicted_class = classes[np.argmax(prediction)]

    response = {"predicted_topic": predicted_class}
    return response
