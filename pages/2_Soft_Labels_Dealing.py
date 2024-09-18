import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.data_loader import DataLoader


# Function to convert hard labels to soft labels
def hard_to_soft_labels(hard_labels, num_classes, smoothing):
    # Create uniformly distributed soft labels
    soft_labels = np.full((len(hard_labels), num_classes), smoothing / (num_classes - 1))
    for i, label in enumerate(hard_labels):
        soft_labels[i][label] = 1.0 - smoothing if num_classes > 1 else 1.0
    return soft_labels

st.title('Deal with your Dataset ')

# If soft labels are not in session_state, initialize as None
if 'soft_labels' not in st.session_state:
    st.session_state['soft_labels'] = None

# Check if file information is in session_state
if 'file_path' in st.session_state and st.session_state['file_path'] is not None:
    st.write(f"File loaded: {st.session_state['file_name']}")
    
    # Read data from session_state
    df = st.session_state['data']
    st.write("Data preview:")
    st.write(df.head())

    # Default to the first column or use the previously selected label column
    default_label = st.session_state.get('label_column', df.columns[0])
    
    # Allow the user to select a column as label
    label_column = st.selectbox("Select the column to be used as label", df.columns, index=list(df.columns).index(default_label))
    
    try:
        # Check if the selected column is numeric
        if not pd.api.types.is_numeric_dtype(df[label_column]):
            raise ValueError("Selected column is not numeric")
        
        # Check if the selected column contains missing values
        if df[label_column].isnull().any():
            raise ValueError("Selected column contains missing values")

        # Save the selected label column to session_state
        st.session_state['label_column'] = label_column
        st.write(f"Selected label column: {st.session_state['label_column']}")

    
    
        # Handle soft labels
        if 'softlabel' not in st.session_state:
            # Set default soft label parameter
            softlabel = st.slider("Adjust soft label parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            st.session_state['softlabel'] = softlabel
        else:
            # Display the previously set soft label parameter
            st.write(f"Soft label parameter: {st.session_state['softlabel']}")
            softlabel = st.slider("Adjust soft label parameter", min_value=0.0, max_value=1.0, value=st.session_state['softlabel'], step=0.1)
            st.session_state['softlabel'] = softlabel
            
        # Generate soft labels based on the selected label column and soft label parameter
        num_classes = df[label_column].nunique()  # Get the number of unique classes
        hard_labels = df[label_column].astype(int).values  # Convert hard labels to integer array
        
        # Handle the case where all labels are the same (only one class)
        if num_classes == 1:
            st.warning("All labels are the same. Soft labels will be a single class.")
            soft_labels = np.ones((len(hard_labels), 1))  # Create soft labels as all 1 for a single class
            soft_label_columns = [f"soft_label_0"]  # Only one column for the single class
        else:
            soft_labels = hard_to_soft_labels(hard_labels, num_classes, softlabel)
            soft_label_columns = [f"soft_label_{i}" for i in range(num_classes)]
        
        # Convert soft_labels to a DataFrame before concatenating
        soft_labels_df = pd.DataFrame(soft_labels, columns=soft_label_columns, index=df.index)
        
        # Drop the original label column and add the soft labels to the DataFrame
        df = df.drop(columns=[label_column])
        df = pd.concat([df, soft_labels_df], axis=1)
        
        # Update the modified DataFrame in session_state
        st.session_state['data_updated'] = df
        
        st.write("Updated DataFrame with soft labels:")
        st.write(st.session_state['data_updated'].head())
        
    
    except ValueError as e:
        st.error(f"Error: {e}")

else:
    st.warning("You haven't uploaded your Dataset yet.")
