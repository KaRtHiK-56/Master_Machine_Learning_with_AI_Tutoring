#importing necessary libararies 
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings,HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st 
from streamlit_option_menu import option_menu 
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

#creating the frontend of the application using streamlit 

with st.sidebar:
    select = option_menu(
        menu_title = "Menu",
        options = ['Home','Tutor'],
        icons = ['house','vector-pen'],
        default_index = 0, 
        orientation = 'horizontal',

    )

if select == 'Home':
    st.title("Welcome")

if select == "Tutor":
    st.title("Master Machine Learning with AI-Tutoring")
    question = st.text_area("Please enter your query or questions to start learning")

    #creating the function definition 
    def machine_learning(question):
        #loading the document 
        doc = PyPDFLoader('ml.pdf')
        doc = doc.load()
        print("loading of the document is done")

        #splitting the data into chunks 
        splitter = RecursiveCharacterTextSplitter(chunk_size = 10000 , chunk_overlap = 1000)
        splitter = splitter.split_documents(doc)
        print("splitting of the document is done")

        #defining the embedding model 
        embeddings = OllamaEmbeddings(model='nomic-embed-text')
        #embeddings = HuggingFaceEmbeddings(
        #    model_name = 'nomic-embed-text',
        #    model_kwargs={'device':'cpu'},
        #    encode_kwargs={'normalize_embeddings':True}
        #)

        #storing the chunked documents as vector embedding in vectorstore 
        print("embedding and storing process started")

        db = Chroma.from_documents(splitter,embeddings)
        print("stored in vector db")

        #defining the lage language model 
        llm = Ollama(model = 'llama3',temperature = 0.02)

        #creating the retriver for retriving the data from the db
        retriver = db.as_retriever()

        #creating the prompt template for the model to act as a tutor
        prompt = """   
                    You are an expert Machine Learning tutor. Your role is to understand and clearly explain any machine learning 
                    concepts or questions asked. Provide detailed answers, real-time examples, and coding examples where applicable. 
                    If a question is asked outside the scope of machine learning, respond with: 
                    "This is beyond the scope of the work given to me."
                    For every machine learning question, follow this structure:
                    Concept Explanation: Provide a clear and concise explanation of the concept.
                    Real-Time Examples: Illustrate the concept with real-world examples.
                    Coding Examples: Include coding snippets or examples to demonstrate the concept in practice along with step by step explanation of the code.
                    
                    question : {question}

        """

        chain = create_stuff_documents_chain(llm,prompt)
        chain = create_retrieval_chain(retriver,chain)
        response = chain.invoke({'question':question})
        print("answer is ",response)
        return response['answer']
    


    #creating a submit button and the function call

    submit = st.button("Ask")

    if submit:
        with st.spinner("Generating Answer...."):
            st.write(machine_learning(question))




