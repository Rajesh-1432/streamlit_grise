import os
import json
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API")


def read_pdf_from_directory(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text


def read_txt_from_directory(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
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


def generate_script(user_input, generative_model, reference_context):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    script_docs = new_db.similarity_search(user_input)

    # Construct the prompt string
    prompt = f"""
    Take the persona of a Test Script Generator for different kinds of SAP modules.
    In the context, test scripts are given as a reference.
    Based on the structure in the references generate a new test script using the context and test case.
    Generate a test script strictly based on the test case and purpose/overview of the context.
    For Fiori Codes based on the test script generated, understand it and return the Fiori Codes from the context.
    Strictly don’t use pre-existing test scripts as a response for a test case.
    Do not generate the same test script for two different test cases.
    
    For Example the Structure of the Test Script:

    {{
        "Intracompany STO": {{
            "Step Description": {{
                "1": "Login to SAP Fiori...",
                "2": "Access the App..."
            }},
            "Expected Results": {{
                "1": "SAP Fiori Launchpad is displayed",
                "2": "Purchase Order for STO..."
            }},
            "Fiori ID": "F0842A"
        }}
    }}

    Output Should be strictly in the form of JSON.

    Reference Context: \n{reference_context}\n
    User Input: \n{user_input}\n
    """

    # Load the QA chain
    chain = load_qa_chain(generative_model, chain_type="stuff")

    # Run chain
    response = chain({"input_documents": script_docs, "question": prompt}, return_only_outputs=True)

    output_text = response.get("output_text", "").strip()

    # Remove code block markers if present
    if output_text.startswith("```json") and output_text.endswith("```"):
        output_text = output_text[7:-3].strip()

    return json.loads(output_text)


def script(uploaded_text, case_description):
    references_directory = "References"
    combined_steps_file = os.path.join(references_directory, "combined_steps.json")
    fiori_codes_file = os.path.join(references_directory, "Fioricodes.json")

    # Load combined_steps.json
    with open(combined_steps_file, 'r') as file:
        reference_context = json.load(file)

    # Load Fioricodes.json and merge
    with open(fiori_codes_file, 'r') as file:
        fiori_codes_context = json.load(file)

    reference_context.update(fiori_codes_context)

    # Read reference docs
    reference_text = read_pdf_from_directory(references_directory)
    reference_text += read_txt_from_directory(references_directory)
    reference_text += read_docx_from_directory(references_directory)

    if reference_text:
        understand_tone_and_language(reference_text)

    query = "Generate Test Scripts by understanding the purpose of the doc uploaded"
    if uploaded_text and query:
        generative_model = initialize_generative_model()
        understand_tone_and_language(uploaded_text + reference_text)
        test_scripts = {}
        test_scripts[case_description] = generate_script(case_description, generative_model, reference_context)
        return test_scripts[case_description]