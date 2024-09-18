import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import tempfile
import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.data_loader import DataLoader


st.title('Please Upload Your Dataset')


# 初始化 session_state
if 'file_path' not in st.session_state:
    st.session_state['file_path'] = None
if 'file_name' not in st.session_state:
    st.session_state['file_name'] = None
if 'data' not in st.session_state:
    st.session_state['data'] = None


uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "out"])


if uploaded_file is not None:
    # 保存文件到临时目录
    file_name = uploaded_file.name
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file_name)
    
    # 将文件存储到临时目录中
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 更新 session_state
    st.session_state['file_path'] = file_path
    st.session_state['file_name'] = file_name  # 保存文件名

    # 读取上传的文件数据并存储到 session_state
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_name.endswith('.out'):
        df = pd.read_csv(file_path, header=None)

    st.session_state['data'] = df
    st.write("Data Preview:")
    st.write(df.head())
    st.write("File uploaded successfully!")

