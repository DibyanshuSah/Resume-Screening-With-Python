# Resume Screening with Python

An NLP-based app that classifies resumes into job categories using text processing and machine learning. Built with Scikit-learn and deployed with Streamlit.

## Features
- Upload a resume in PDF, DOCX, or TXT format  
- Extract and preprocess text (tokenization, stopword removal, lemmatization)  
- Convert text into numerical features using TF-IDF  
- Classify resumes into predefined job categories with ML models  
- Simple and interactive UI with Streamlit  

## Tech Stack
Python, NumPy, Pandas, NLTK, Scikit-learn, Seaborn, Streamlit  

## How to Run

:: Clone the repository
```bash
git clone https://github.com/DibyanshuSah/Resume-Screening-With-Python.git
cd Resume-Screening-With-Python
```
:: Create virtual environment
```bash
python -m venv venv
```
:: Activate virtual environment
venv\Scripts\activate
```bash
:: Install dependencies
pip install -r requirements.txt
```
:: Run the Streamlit app
```bash
streamlit run app.py
```
