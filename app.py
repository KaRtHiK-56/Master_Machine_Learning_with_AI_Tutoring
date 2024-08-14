# importing necessary libararies
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from datetime import datetime
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# loading the document
doc = PyPDFLoader("ml.pdf")
doc = doc.load()
print("loading of the document is done")
# splitting the data into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
splitter = splitter.split_documents(doc)
print("splitting of the document is done")
# defining the embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# embeddings = HuggingFaceEmbeddings(
#    model_name = 'nomic-embed-text',
#    model_kwargs={'device':'cpu'},
#    encode_kwargs={'normalize_embeddings':True}
# )
# storing the chunked documents as vector embedding in vectorstore
print("embedding and storing process started")
# logging the time
start_time = datetime.now()
print("starting time is:", start_time)
print("Started to log the time for embedding and storing process")
db = Chroma.from_documents(splitter, embeddings)
end_time = datetime.now()
print("ending time is:", end_time)
print("Duration: {}".format(end_time - start_time))
print("stored in vector db")
# defining the lage language model
llm = Ollama(model="llama3", temperature=0.02)

# creating the prompt template for the model to act as a tutor
prompt = """   
            You are an expert Machine Learning tutor. Your role is to understand and clearly explain any machine learning 
            concepts or questions asked. Provide detailed answers, real-time examples, and coding examples where applicable. 
            If a question is asked outside the scope of machine learning, respond with: 
            "This is beyond the scope of the work given to me."
            For every machine learning question, follow this structure:
            Concept Explanation: Provide a clear and concise explanation of the concept.
            Real-Time Examples: Illustrate the concept with real-world examples.
            Coding Examples: Include coding snippets or examples to demonstrate the concept in practice along with step by step explanation of the code.
            context : {context}
            question : {question}
"""
# creating the prompt template
prompt = PromptTemplate.from_template(template=prompt)
# creating the retriver for retriving the data from the db
retriever = db.as_retriever()
chains = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
question = "what is underfitting in machine learning"
response = chains.invoke(question)
print("answer is ", response)
