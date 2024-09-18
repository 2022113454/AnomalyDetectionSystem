import streamlit as st
import pandas as pd
import os
from collections import Counter
import torch
from matplotlib import pyplot as plt
from utils.timeseries_dataset import read_files, create_splits
from utils.train_deep_model_utils import json_file
from utils.evaluator import Evaluator
from utils.config import deep_models

# Default data path setting
if 'data_path' not in st.session_state:
    st.session_state['data_path'] = "/mnt/data/user3/MSAD/data/TSB_128/Daphnet"

def eval_deep_model_app(fnames, model_name='convnet', model_path=None, model_parameters_file=None):
    # Set default paths if not provided
    if model_path is None:
        model_path = "/mnt/data/user3/MSAD/results/weights/supervised/convnet_default_512/model_30012023_173428"
    if model_parameters_file is None:
        model_parameters_file = "/mnt/data/user3/MSAD/models/configuration/convnet_default.json"
    
    # Try loading the model
    try:
        if os.path.exists(model_parameters_file):
            model_parameters = json_file(model_parameters_file)
            model = deep_models[model_name](**model_parameters)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.to('cuda')
        else:
            raise OSError(f"Local model config not found at {model_parameters_file}.")
    except Exception as e:
        st.error(f"Error loading model: {e}.")
        return

    # Get the data path
    data_path = st.session_state['data_path']

    # Full file paths for evaluation
    fnames_full_path = [os.path.join(data_path, fname) for fname in fnames]

    # Check if files exist
    missing_files = [file for file in fnames_full_path if not os.path.isfile(file)]
    if missing_files:
        st.error(f"Missing files: {missing_files}")
        return

    # Set up progress bar
    progress_bar = st.progress(0)
    total_files = len(fnames_full_path)

    # Evaluate model
    try:
        evaluator = Evaluator()
        results = []

        # Process files and update the progress bar
        for idx, fname in enumerate(fnames_full_path):
            result = evaluator.predict(
                model=model,
                fnames=[fname],  # Use full path for each file
                data_path=data_path,
                deep_model=True,
            )
            results.append(result)

            # Update progress bar
            progress_bar.progress((idx + 1) / total_files)

        # Concatenate all results
        results_df = pd.concat(results)

        # Display a success message
        st.success("Model evaluation completed successfully!")

        # Display summarized results as a dataframe
        st.markdown("### Evaluation Results (Summarized):")
        st.dataframe(results_df.head())  # Show only the first few rows

        # Count and display class occurrences in a table format
        counter = Counter(results_df["class"])
        st.markdown("### Class Distribution:")
        st.table(pd.DataFrame(counter.items(), columns=["Class", "Count"]))

        # Plot class distribution as a bar chart
        st.markdown("### Class Distribution Bar Chart:")
        fig, ax = plt.subplots()
        ax.bar(counter.keys(), counter.values(), color='skyblue')
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during model evaluation: {e}")

# Streamlit page setup
st.title("Deep Learning Model Evaluation")
st.markdown("""
Welcome to the **Deep Learning Model Evaluation Tool**. Please upload a CSV file containing the list of time-series filenames you wish to evaluate. 
Ensure the CSV contains a single column named `filename`.
""")
st.write("---")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        data = pd.read_csv(uploaded_file)
        if 'filename' not in data.columns:
            st.error("The uploaded CSV file must contain a 'filename' column.")
        else:
            # Extract filenames and pass to the evaluation function
            fnames = data['filename'].tolist()
            st.markdown(f"**{len(fnames)} files found**. Starting evaluation...")
            eval_deep_model_app(fnames)
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
else:
    st.info("Please upload a CSV file to start the evaluation.")
