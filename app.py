import os
import glob
import zipfile
import streamlit as st
import pandas as pd
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from docx import Document
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from testscripts import script
import base64

# Initialize environment and API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API")

# === UI Styling (Black & White Only) ===
st.markdown(
    """
    <style>
    /* General app background */
    .stApp {
        background-color: #ffffff !important; /* white background */
        color: #000000 !important; /* black text */
    }

    /* Ensure all main text elements are black */
    .stApp p, .stApp span, .stApp div, .stApp label {
        color: #000000 !important;
    }

    /* Title */
    .center-title {
        text-align: center;
        font-size: 2em;
        font-weight: 600;
        color: #000000 !important;
        margin-bottom: 25px;
    }

    /* Primary buttons - ensure white text on black background */
    button, form_submit_button, div.stButton > button, .stFileUploader button, [data-baseweb="button"] {
        background-color: #000000 !important; /* Black background */
        color: #ffffff !important;           /* White text */
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        transition: background-color 0.3s, transform 0.2s;
    }

    /* Form submit buttons - specifically target them */
    .stForm button, .stForm form_submit_button, .stForm button span {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    /* Hover effect */
    button:hover, div.stButton > button:hover, .stFileUploader button:hover, [data-baseweb="button"]:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
        transform: translateY(-2px);
    }

    /* Download button */
    .download-button {
        display: block;
        width: 250px;
        margin: 25px auto;
        padding: 10px 20px;
        color: #ffffff !important;
        background-color: #000000 !important;
        text-align: center;
        border-radius: 6px;
        text-decoration: none;
        font-size: 1.05em;
        font-weight: 600;
        transition: background-color 0.3s, transform 0.2s;
    }

    .download-button:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
        transform: translateY(-2px);
        text-decoration: none;
    }

    /* Radio buttons & file uploader label */
    .stRadio > div{
        color: #000000 !important;
    }

    .stFileUploader > label {
        color: #ffffff !important;
    }

    /* Alerts */
    .stAlert {
        color: #000000 !important;
    }

    /* Ensure button text is always white - more specific selectors */
    button span, div.stButton > button span, .stForm button span {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Functions ===
def scrolling_headline(state, placeholder, headline_text: str):
    if state:
        html_code = f"""
        <div style="overflow: hidden; white-space: nowrap; width: 100%; box-sizing: border-box;">
            <marquee scrollamount="5" behavior="scroll" direction="right" style="font-size:1em;color:#000000;font-weight:bold;">
                {headline_text}
            </marquee>
        </div>
        """
        placeholder.markdown(html_code, unsafe_allow_html=True)
    else:
        placeholder.success(headline_text)


def read_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def read_txt(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.read().decode("utf-8")
    return text


def read_docx(docx_docs):
    text = ""
    for docx in docx_docs:
        doc = Document(docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text


def read_pdf_from_directory(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text


def read_txt_from_directory(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            txt_path = os.path.join(directory, filename)
            with open(txt_path, 'r', encoding='utf-8') as file:
                text += file.read() + "\n"
    return text


def read_docx_from_directory(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith('.docx'):
            docx_path = os.path.join(directory, filename)
            doc = Document(docx_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
    return text


def initialize_generative_model():
    return ChatOpenAI(model="gpt-4o", temperature=0.5)


def understand_tone_and_language(script_text):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_texts([script_text], embedding=embeddings)
    vector_store.save_local("faiss_index")


def generate_script(user_input, generative_model):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    script_docs = new_db.similarity_search(user_input)

    prompt_template = """
    Take the persona of AI SAP Test Case Generator.
    A Testcase is an end to end workflow but not a single step process in SAP.
    Extract the overview/purpose of the input provided.
    For the overview extracted generate the test case/s.
    Do not generate test scripts or descriptions.
    Each test case should not exceed 10 words.
    Format: "Test Case X: [Description]"
    Provide at least 5 test cases.
    Context: \n{context}\n
    Generated Test Cases:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(generative_model, chain_type="stuff", prompt=prompt)

    response = chain({"input_documents": script_docs}, return_only_outputs=True)
    return response["output_text"]


def test_cases_to_excel(test_cases, file_name="test_cases.xlsx"):
    data = []
    for case in test_cases.splitlines():
        if case.startswith("Test Case"):
            if ": " in case:
                description = case.split(": ", 1)[1].strip()
                test_case_number = case.split(": ", 1)[0].strip()
                data.append([test_case_number, description])
            else:
                data.append([case.strip(), ""])
    df = pd.DataFrame(data, columns=["Test Case Number", "Description"])
    return df


def convert_to_excel(test_scripts, file_path):
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        summary_sheet_name = "Test Cases"
        summary_data = []
        workbook = writer.book
        header_format = workbook.add_format({'bold': True, 'bg_color': '#DDDDDD'})  # Gray header
        title_format = workbook.add_format({'bold': True, 'bg_color': '#FFFFFF', 'font_color': '#000000'})

        for i, (case_title, script_details) in enumerate(test_scripts.items(), start=1):
            steps = []
            for scenario, details in script_details.items():
                fiori_id = details.get("Fiori ID", "")
                step_desc = details.get("Step Description", {})
                exp_results = details.get("Expected Results", {})
                for step_num, step_desc_text in step_desc.items():
                    steps.append({
                        "Step Number": step_num,
                        "Step Description": step_desc_text,
                        "Expected Result": exp_results.get(step_num, ""),
                        "Fiori ID": fiori_id,
                    })

            df = pd.DataFrame(steps)
            sheet_name = f"TestScript {i}"
            df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=1)
            worksheet = writer.sheets[sheet_name]
            worksheet.write(0, 0, case_title, title_format)
            summary_data.append([f"Test Case {i}: {case_title}"])
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(1, col_num, value, header_format)

        summary_df = pd.DataFrame(summary_data, columns=["Test Case Overview"])
        summary_df.to_excel(writer, index=False, sheet_name=summary_sheet_name, startrow=0)
        summary_worksheet = writer.sheets[summary_sheet_name]
        summary_worksheet.set_column(0, 0, 50)
        summary_worksheet.write(0, 0, "Test Case Overview", header_format)


def process_file(uploaded_file, reference_text, output_directory, base_name=None):
    file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else base_name
    base_name = base_name or os.path.splitext(file_name)[0]
    headline_text = f"Generating Test Scripts for {base_name}"
    placeholder = st.empty()
    scrolling_headline(True, placeholder, headline_text)

    uploaded_text = ""
    if file_name.endswith(".pdf"):
        uploaded_text += read_pdf([uploaded_file])
    elif file_name.endswith(".txt"):
        uploaded_text += read_txt([uploaded_file])
    elif file_name.endswith(".docx"):
        uploaded_text += read_docx([uploaded_file])

    if not uploaded_text:
        st.warning(f"Skipped empty or unsupported file: {file_name}")
        return None

    combined_text = uploaded_text + reference_text
    understand_tone_and_language(combined_text)
    generative_model = initialize_generative_model()
    response = generate_script(uploaded_text, generative_model)
    df = test_cases_to_excel(response)

    testscript = {}
    for j in df["Description"]:
        testscript[j] = script(uploaded_text, j)

    excel_path = os.path.join(output_directory, f"{base_name}.xlsx")
    end_text = f"Test Script for {base_name} is generated"
    scrolling_headline(False, placeholder, end_text)
    convert_to_excel(testscript, excel_path)
    return excel_path


def main():
    st.markdown("<div class='center-title'>Test Script Generator</div>", unsafe_allow_html=True)

    input_method = st.radio("Choose input method:", ("Upload File", "Upload ZIP File"))
    references_directory = "References"
    output_directory = "Generated Excels"
    os.makedirs(output_directory, exist_ok=True)

    reference_text = ""
    reference_text += read_pdf_from_directory(references_directory)
    reference_text += read_txt_from_directory(references_directory)
    reference_text += read_docx_from_directory(references_directory)

    if reference_text:
        understand_tone_and_language(reference_text)

    if input_method == "Upload File":
        uploaded_files = st.file_uploader(" ", type=['pdf', 'txt', 'docx'], accept_multiple_files=True)
        if uploaded_files:
            with st.form(key="generate_form"):
                submit_button = st.form_submit_button("Generate Test Scripts")
                if submit_button:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        if len(uploaded_files) == 1:
                            uploaded_file = uploaded_files[0]
                            file_base_name = os.path.splitext(uploaded_file.name)[0]
                            excel_path = process_file(uploaded_file, reference_text, temp_dir, file_base_name)
                            if excel_path:
                                with open(excel_path, 'rb') as f:
                                    file_data = f.read()
                                    b64 = base64.b64encode(file_data).decode()
                                st.markdown(
                                    f"""
                                    <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" 
                                    download="{os.path.basename(excel_path)}" class="download-button">
                                    Download {os.path.basename(excel_path)}
                                    </a>
                                    """,
                                    unsafe_allow_html=True
                                )
                            else:
                                st.warning("Failed to generate the test script.")
                        else:
                            generated_excel_paths = []
                            for uploaded_file in uploaded_files:
                                file_base_name = os.path.splitext(uploaded_file.name)[0]
                                excel_path = process_file(uploaded_file, reference_text, temp_dir, file_base_name)
                                if excel_path:
                                    generated_excel_paths.append(excel_path)

                            if generated_excel_paths:
                                zip_file_path = os.path.join(temp_dir, "Generated_Test_Scripts.zip")
                                with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                                    for file_path in generated_excel_paths:
                                        zipf.write(file_path, os.path.basename(file_path))
                                with open(zip_file_path, 'rb') as f:
                                    zip_data = f.read()
                                    zip_b64 = base64.b64encode(zip_data).decode()
                                st.markdown(
                                    f"""
                                    <a href="data:application/zip;base64,{zip_b64}" 
                                    download="Generated_Test_Scripts.zip" class="download-button">
                                    Download All Generated Test Scripts
                                    </a>
                                    """,
                                    unsafe_allow_html=True
                                )
                            else:
                                st.warning("No test scripts were generated.")

    elif input_method == "Upload ZIP File":
        uploaded_zip = st.file_uploader(" ", type='zip')
        if uploaded_zip:
            if st.button("Generate Test Scripts"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                    supported_extensions = ('*.pdf', '*.txt', '*.docx')
                    uploaded_files = []
                    for ext in supported_extensions:
                        uploaded_files.extend(glob.glob(os.path.join(temp_dir, ext)))

                    if uploaded_files:
                        generated_excel_paths = []
                        for uploaded_file in uploaded_files:
                            file_base_name = os.path.splitext(os.path.basename(uploaded_file))[0]
                            with open(uploaded_file, 'rb') as f:
                                excel_path = process_file(f, reference_text, output_directory, file_base_name)
                                if excel_path:
                                    generated_excel_paths.append(excel_path)
                        if generated_excel_paths:
                            original_zip_name = os.path.splitext(uploaded_zip.name)[0]
                            zip_file_path = os.path.join(temp_dir, f"{original_zip_name}_Generated_Test_Scripts.zip")
                            with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                                for file_path in generated_excel_paths:
                                    zipf.write(file_path, os.path.basename(file_path))
                            with open(zip_file_path, 'rb') as f:
                                zip_data = f.read()
                                zip_b64 = base64.b64encode(zip_data).decode()
                            st.markdown(
                                f"""
                                <a href="data:application/zip;base64,{zip_b64}" 
                                download="{original_zip_name}_Generated_Test_Scripts.zip" class="download-button">
                                Download All Generated Test Scripts
                                </a>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning("No test scripts were generated.")
                    else:
                        st.warning("No supported files found in the ZIP file.")


if __name__ == "__main__":
    main()