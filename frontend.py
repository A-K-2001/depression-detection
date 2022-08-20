import streamlit as st
from tensorflow import keras
from transformers import AutoTokenizer, TFAutoModel
import numpy as np

max_length: int = 30
model_name: str = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = TFAutoModel.from_pretrained(model_name)
model = keras.models.load_model(
    "./model.h5", custom_objects={"TFBertModel": bert}
)

def check(url):
    tokens = tokenizer.encode_plus(
        url,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="tf",
    )
    result = model.predict([tokens["input_ids"], tokens["attention_mask"]])
    print(result,"**********************************************")
    return np.argmax(result)

st.title('Depression Detector')
url = st.text_input("Enter Tweet")

if st.button("Check"):
    res=check(url)
    if res==1:
        st.write("Depression detected")
    else:
        st.write("No Depression detected")