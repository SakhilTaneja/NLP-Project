#NLP-Project
##This repository contains code for Project: Automation of Identification and Summarization of Successful cases, This repository does not contains the data used in this project.

##Notebook sub-repository contains code of Data preprocessing, Feature extraction, Building Classsification Model, Data Analysis i.e. The complete process of building and choosing classification model.

##App sub-repository contains code used in final build:
### - Data Preprocessing, feature extraction and our classification model in classify_model.py
### - Summarization model integration using langchain in summary_model.py module
### - Front End using Streamlit in app.py
### - Dockerfile and requirements.txt containing info for building our container and environment using docker
### - text_classifier.pkl is our final classification model extracted after data analysins using Joblib

##Summarization Model Used:- llama-2-7b-chat.03_K_L.gguf, Steps to download the summarization model:
### 1) go to https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
### 2) go to 'Files and versions'
### 3) download 'llama-2-7b-chat.Q3_K_L.gguf' from it
### 4) keep it in the app folder

