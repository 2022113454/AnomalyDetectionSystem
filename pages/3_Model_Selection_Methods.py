import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import utils.data_loader as dl

while True:
    input_text = st.text_input('Input your file path here')
    if input_text:
        break
dataloader = dl.DataLoader(input_text)
datasets = dataloader.get_dataset_names()
x,y,frames = dataloader.load(datasets)
st.write(x,y,frames)
