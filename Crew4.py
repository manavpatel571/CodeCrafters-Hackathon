import asyncio
from langchain import ChatGoogleGenerativeAI, GooglePalmEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
import tempfile

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
