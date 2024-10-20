import streamlit as st
import requests
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import os
from dotenv import load_dotenv

# ------------------------------
# Load Environment Variables
# ------------------------------
load_dotenv()

# ------------------------------
# Configuration
# ------------------------------
LLAMA_TEXT_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLAMA_IMAGE_API_URL = "https://api.groq.com/llama3/image"
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error("API Key not found. Please set the GROQ_API_KEY environment variable.")
    st.stop()

# ------------------------------
# Helper Functions
# ------------------------------

def get_llama_text_response(text, model="llama-3.2-chat"):
    """
    Sends text input to the Llama 3 API and returns the response.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": text}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
    try:
        response = requests.post(LLAMA_TEXT_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response received.")
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err} - {response.text}"
    except Exception as err:
        return f"An error occurred: {err}"

def get_llama_image_response(image_bytes, model="llama-3.2-image"):
    """
    Sends image input to the Llama 3 API and returns the response.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}"
        # 'Content-Type' is set automatically by 'requests' when using 'files'
    }
    files = {
        "image_file": ("image.png", image_bytes, "image/png")
    }
    try:
        response = requests.post(LLAMA_IMAGE_API_URL, files=files, headers=headers)
        response.raise_for_status()
        return response.json().get("response", "No response received.")
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err} - {response.text}"
    except Exception as err:
        return f"An error occurred: {err}"

def analyze_network_data(data, model="llama-3.2-analysis"):
    """
    Analyzes network performance data using the Llama 3 API.
    """
    summary = data.describe().to_json()
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"Analyze the following network performance data summary:\n{summary}"}
        ],
        "max_tokens": 200,
        "temperature": 0.5
    }
    try:
        response = requests.post(LLAMA_TEXT_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No analysis received.")
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err} - {response.text}"
    except Exception as err:
        return f"An error occurred: {err}"

# ------------------------------
# Streamlit App Layout
# ------------------------------

def main():
    st.set_page_config(page_title="ConnectAssist", layout="wide")
    st.sidebar.title("ConnectAssist")
    app_mode = st.sidebar.selectbox("Choose a Feature",
                                    ["Home", "Troubleshooting", "Training", "Network Insights", "Connectivity Solutions"])

    if app_mode == "Home":
        home()
    elif app_mode == "Troubleshooting":
        troubleshooting_assistant()
    elif app_mode == "Training":
        training_modules()
    elif app_mode == "Network Insights":
        network_insights()
    elif app_mode == "Connectivity Solutions":
        connectivity_solutions()

# ------------------------------
# Home Page
# ------------------------------

def home():
    st.title("Welcome to ConnectAssist")
    st.write("""
    **ConnectAssist** is an AI-powered mobile application designed to empower telecommunications field workers operating in underserved regions. Leveraging Llama 3.2’s lightweight and multimodal capabilities, ConnectAssist provides real-time support, diagnostics, and training to enhance network deployment and maintenance efficiency.
    """)
    st.image("https://via.placeholder.com/800x400.png?text=ConnectAssist+Event+Thumbnail", use_column_width=True)

# ------------------------------
# Troubleshooting Assistant
# ------------------------------

def troubleshooting_assistant():
    st.header("Real-Time Troubleshooting Assistant")
    
    st.subheader("Describe the Issue")
    issue_text = st.text_area("Enter a description of the problem:", height=150)
    
    if st.button("Get Suggestions"):
        if issue_text:
            with st.spinner("Processing..."):
                response = get_llama_text_response(issue_text)
            if response.startswith("Error"):
                st.error(response)
            else:
                st.success("**Diagnostic Suggestions:**")
                st.write(response)
        else:
            st.warning("Please enter a description of the issue.")
    
    st.markdown("---")
    
    st.subheader("Upload an Image of the Issue")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                image_bytes = uploaded_image.read()
                response = get_llama_image_response(image_bytes)
            if response.startswith("Error"):
                st.error(response)
            else:
                st.success("**Image Analysis:**")
                st.write(response)

# ------------------------------
# Training Modules
# ------------------------------

def training_modules():
    st.header("Dynamic Training Modules")
    
    st.subheader("Interactive Tutorials")
    st.markdown("""
    - **Module 1:** Setting Up Network Equipment
    - **Module 2:** Troubleshooting Common Issues
    - **Module 3:** Optimizing Network Performance
    """)
    
    if st.button("Watch Tutorial 1"):
        # Replace with the actual video URL or local file path
        st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    
    st.markdown("---")
    
    st.subheader("Augmented Reality Guides")
    st.markdown("""
    While AR integration is beyond the scope of this prototype, here's a simulated guide:
    """)
    st.image("https://via.placeholder.com/800x400.png?text=Simulated+AR+Guide", caption="Simulated AR Guide", use_column_width=True)

# ------------------------------
# Network Optimization Insights
# ------------------------------

def network_insights():
    st.header("Network Optimization Insights")
    
    st.subheader("Upload Network Performance Data")
    uploaded_file = st.file_uploader("Choose a CSV file...", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("**Data Preview:**")
            st.dataframe(data.head())
            
            st.subheader("Performance Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'signal_strength' in data.columns:
                    st.line_chart(data['signal_strength'])
                else:
                    st.write("**signal_strength** column not found.")
            
            with col2:
                if 'bandwidth_usage' in data.columns:
                    st.bar_chart(data['bandwidth_usage'])
                else:
                    st.write("**bandwidth_usage** column not found.")
            
            if st.button("Analyze Data"):
                with st.spinner("Analyzing data..."):
                    analysis = analyze_network_data(data)
                if analysis.startswith("Error"):
                    st.error(analysis)
                else:
                    st.success("**Analysis Results:**")
                    st.write(analysis)
        except Exception as e:
            st.error(f"Error processing the file: {e}")

# ------------------------------
# Connectivity Solutions
# ------------------------------

def connectivity_solutions():
    st.header("Localized Connectivity Solutions")
    
    st.subheader("User Reports")
    report_text = st.text_area("Enter user reports or observations:", height=150)
    
    st.subheader("Site Photos")
    uploaded_image = st.file_uploader("Upload site photos...", type=["jpg", "png", "jpeg"])
    
    if st.button("Generate Recommendations"):
        if report_text and uploaded_image:
            with st.spinner("Generating recommendations..."):
                text_response = get_llama_text_response(report_text)
                image_bytes = uploaded_image.read()
                image_response = get_llama_image_response(image_bytes)
                combined_recommendation = f"{text_response}\n\n{image_response}"
            if combined_recommendation.startswith("Error"):
                st.error(combined_recommendation)
            else:
                st.success("**Recommendations:**")
                st.write(combined_recommendation)
        else:
            st.warning("Please provide both user reports and site photos.")

# ------------------------------
# Run the App
# ------------------------------

if __name__ == "__main__":
    main()
