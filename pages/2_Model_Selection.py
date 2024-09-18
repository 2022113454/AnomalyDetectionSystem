import streamlit as st 

st.write("Choose a model")
model_list = ["avg_ens", "convnet", "resnet", "oracle", "sit", "inception_time"]
if 'model' not in st.session_state:
    st.session_state['model'] = None
    model = st.selectbox(
        "Please select a model:",
        ["avg_ens", "convnet", "resnet", "oracle", "sit", "inception_time"]
    )
    st.session_state['model'] = model
else:
    default_model = st.session_state['model']
    model = st.selectbox(
        "Please select a model:",
        ["avg_ens", "convnet", "resnet", "oracle", "sit", "inception_time"],index = model_list.index(default_model)
    )
    st.session_state['model'] = model




