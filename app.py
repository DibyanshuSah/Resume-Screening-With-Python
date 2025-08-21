import streamlit as st
import pickle
import docx
import PyPDF2
import re
import time

# Category mapping dictionary
category_mapping = {
    15: "JAVA DEVELOPER",
    23: "TESTING",
    8: "DEVOPS ENGINEER",
    20: "PYTHON DEVELOPER",
    24: "WEB DESIGNING",
    12: "HR",
    13: "HADOOP",
    22: "SALES",
    6: "DATA SCIENCE",
    16: "MECHANICAL ENGINEER",
    10: "ETL DEVELOPER",
    3: "BLOCKCHAIN",
    18: "OPERATIONS MANAGER",
    1: "ARTS",
    7: "DATABASE",
    14: "HEALTH AND FITNESS",
    19: "PMO",
    11: "ELECTRICAL ENGINEERING",
    4: "BUSINESS ANALYST",
    9: "DOTNET DEVELOPER",
    2: "AUTOMATION TESTING",
    17: "NETWORK SECURITY ENGINEER",
    5: "CIVIL ENGINEER",
    21: "SAP DEVELOPER",
    0: "ADVOCATE"
}

# Load model and vectorizer
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText.strip()

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = svc_model.predict(vectorized_text)[0]
    return category_mapping.get(predicted_category, "Unknown Category")

def main():
    st.set_page_config(page_title="CVisionAI – Smart screening, sharper hiring.", page_icon="📄", layout="wide")

    # Center all app content vertically and horizontally with flexbox
    st.markdown(
        """
        <style>
        .appview-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 90vh;
            flex-direction: column;
            text-align: center;
        }
        .stFileUploader > div {
            max-width: 400px;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='color: #1E90FF;'>🔍 CVisionAI – Smarter screening, faster hiring.</h1>", unsafe_allow_html=True)
    st.markdown("<h4>Upload a resume in PDF, DOCX, or TXT format to get the predicted job category.</h4>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("Text extracted from resume successfully!")

            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            st.subheader("Predicted Category")

            # Spinner with delays for processing messages
            with st.spinner("Processing your resume..."):
                time.sleep(1.5)
            with st.spinner("Analyzing the best match..."):
                time.sleep(1.5)

            category = pred(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
