from flask import Flask, request, jsonify, render_template
from Crew1 import start_crew as start_crew1
from Crew2 import start_crew as start_crew2
from Crew3 import start_crew3

import markdown
import os
import tempfile
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Load the Google API Key
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/main.html')
def main():
    return render_template('main.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/services.html')
def services():
    return render_template('services.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/start', methods=['POST'])
def start():
    crew_type = request.form.get('crew_type')
    print(f"Received crew_type: {crew_type}")  # Debugging print statement

    if crew_type == 'crew1':
        topic = request.form.get('topic')
        if not topic:
            return jsonify({"error": "Topic is required for Service 1"}), 400
        result = start_crew1(topic)
    elif crew_type == 'crew2':
        topic = request.form.get('topic')
        if not topic:
            return jsonify({"error": "Topic is required for Service 2"}), 400
        result = start_crew2(topic)
    elif crew_type == 'crew3':
        participants = request.form.get('participants')
        context = request.form.get('context')
        objective = request.form.get('objective')
        if not participants or not context or not objective:
            return jsonify({"error": "Participants, Context, and Objective are required for Service 3"}), 400
        result = start_crew3(participants, context, objective)
    elif crew_type == 'crew4':
        if 'pdf_files' not in request.files:
            return jsonify({"error": "PDF files are required for Service 4"}), 400
        pdf_files = request.files.getlist('pdf_files')
        question = request.form.get('question')
        if not question:
            return jsonify({"error": "Question is required for Service 4"}), 400

        # Process PDF files and generate embeddings
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        embeddings, llm, prompt = loop.run_until_complete(initialize_service4(pdf_files))

        # Perform the search and get the response
        response = perform_qa(question, embeddings, llm, prompt)

        result = response['answer']
    else:
        return jsonify({"error": "Invalid crew type received: " + str(crew_type)}), 400

    html_output = markdown.markdown(result).replace('\n', '<br>')
    return jsonify({"result": html_output})

async def initialize_service4(pdf_files):
    llm = ChatGoogleGenerativeAI(model='gemini-pro')
    embeddings = GooglePalmEmbeddings()
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide a detailed and thorough response based on the question.
        Also provide relevant papers from the research PDFs.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    all_documents = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        all_documents.extend(final_documents)

    vectors = FAISS.from_documents(all_documents, embeddings)
    vectors.save_local('faiss_index')

    return vectors, llm, prompt

def perform_qa(question, embeddings, llm, prompt):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = embeddings.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': question})
    return response

if __name__ == '__main__':
    app.run(debug=True)
