import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
MAX_LEN = 60
st.set_page_config(page_title="YouTube Sentiment Analyzer", page_icon="🎬")
# ---------- LOAD FILES ----------
@st.cache_resource #@st.cache_resource is a Streamlit decorator used to cache heavy objects like:Machine learning models,Tokenizers,Large files
#Large files.It tells Streamlit:“Load this only once. Don’t reload it every time the app reruns.”
def load_model():
     model = tf.keras.models.load_model("final_ann_youtube_sentiment.h5")
     return model
@st.cache_resource
def load_tokenizer():
     with open("tokenizer.pkl", "rb") as f:
         return pickle.load(f)
@st.cache_resource
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)
with st.spinner("Loading AI Model... Please wait ⏳"):
    model = load_model()
    tokenizer=load_tokenizer()
    le=load_label_encoder()
# ---------- TEXT CLEANING ----------
def clean_text(t):
    t = t.lower()
    t = re.sub(r"https\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return t
# ---------- PREDICTION ----------
def predict_sentiment(text):
    cleaned = clean_text(text)
    seq=tokenizer.texts_to_sequences([cleaned])
    padded=pad_sequences(seq,maxlen=MAX_LEN,padding='post')
    probs = model.predict(padded)[0]
    class_index = np.argmax(probs)
    label=le.inverse_transform([class_index])[0]
    confidence = round(float(np.max(probs)) * 100, 2)
    return label, confidence
# ---------- UI ----------
st.title("🎬 YouTube Comment Sentiment Analyzer")
st.markdown("Analyze YouTube comments using a trained **ANN Deep Learning Model**")
user_input=st.text_area("✍️ Enter a YouTube comment:")
if st.button("Analyze Sentiment"):
    #edge case
    if user_input.strip()=="":
        st.warning("Please enter a comment.")
    else:
        label, confidence = predict_sentiment(user_input)
        if label.lower() == "positive":
            st.success(f"😊 Positive Sentiment ({confidence}%)")
        elif label.lower() == "negative":
            st.error(f"😠 Negative Sentiment ({confidence}%)")
        else:
            st.info(f"😐 Neutral Sentiment ({confidence}%)")
st.markdown("---")
st.markdown("Built with ❤️ using ANN,Simple RNN & BiLSTM | Streamlit | TensorFlow")
        