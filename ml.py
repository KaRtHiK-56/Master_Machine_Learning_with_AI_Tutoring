#importing necessary libararies 
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st 
from streamlit_option_menu import option_menu 


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

    #loading the document 
    doc = PyPDFLoader('book.pdf')
    doc = doc.load()
    print("loading of the document is done")

    #defining the embedding model 
    embeddings = OllamaEmbeddings(model='mxbai-embed-large:335m')
    
    #splitting the data into chunks 
    splitter = RecursiveCharacterTextSplitter(chunk_size = 50000 , chunk_overlap = 10000)
    splitter = splitter.split_documents(doc)
    print("splitting of the document is done")

    #storing the chunked documents as vector embedding in vectorstore 
    print("embedding and storing process started")
    
    db = Chroma.from_documents(splitter,embeddings)
    print("stored in vector db")

    #defining the lage language model 
    llm = Ollama(model = 'llama3',temperature = 0.02)

    #creating the retriver for retriving the data from the db
    retriver = db.as_retriever()

    #creating the function definition 
    def machine_learning(question):

        #creating the prompt template for the model to act as a tutor
        prompter = """   
                    You are an expert Machine Learning tutor. Your role is to understand and clearly explain any machine learning 
                    concepts or questions asked. Provide detailed answers, real-time examples, and coding examples where applicable. 
                    If a question is asked outside the scope of machine learning, respond with: 
                    "This is beyond the scope of the work given to me."
                    For every machine learning question, follow this structure:
                    Concept Explanation: Provide a clear and concise explanation of the concept.
                    Real-Time Examples: Illustrate the concept with real-world examples.
                    Coding Examples: Include coding snippets or examples to demonstrate the concept in practice along with step by step explanation of the code.
                    <context>
                    {context}
                    </context>
                    question : {question}

                    Example interaction:

                    User: What is overfitting in machine learning?
                    AI Tutor:

                    Concept Explanation: Overfitting occurs when a machine learning model learns the details and noise in the training data to the extent that it negatively impacts the model's performance on new data. This means the model performs well on training data but poorly on unseen data.
                    Real-Time Example: Imagine a student who memorizes answers to questions for a test rather than understanding the underlying concepts. They score highly on that specific test but struggle with any variation in the questions.
                    Coding Example:
                    from sklearn.model_selection import train_test_split
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import mean_squared_error

                    # Example data
                    X = [[i] for i in range(10)]
                    y = [2*i + 1 for i in range(10)]

                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train the model
                    model = LinearRegression()
                    model.fit(X_train, y_train)

                    # Predict and evaluate
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

                    # Mean squared error
                    print(f'Training MSE: {mean_squared_error(y_train, y_train_pred)}')
                    print(f'Testing MSE: {mean_squared_error(y_test, y_test_pred)}')


                    User: Can you explain the theory behind blockchain?
                    AI Tutor: This is beyond the scope of the work given to me.
        """

        prompt_template = PromptTemplate.from_template(template = prompter)
        prompt = prompt_template.format(prompter = prompter)
        chain = prompt | llm |retriver
        response = chain.invoke({'question':question})
        print("answer is ",response)
        return response['answer']
    


    #creating a submit button and the function call

    submit = st.button("Ask")

    if submit:
        with st.spinner("Generating Answer...."):
            st.write(machine_learning(question))




