import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
import os

logger = logging.getLogger(__name__)

openai_api_key= os.getenv("OPENAI_API_KEY")

llm= ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.2)

parser = StrOutputParser()


def answer_code_question(user_message, code_context):
    
    prompt_template = """
    Answer the question as detailed as possible from the provided context(Python code Explanation), 
    make sure to provide all the details, if the answer is not in provided context just say, 
    "answer is not available in the context", don't provide the wrong answer.
    
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    
    prompt = ChatPromptTemplate([('user', prompt_template)])
    
    chain = prompt | llm | parser
    
    response = chain.invoke({'context': code_context, 'question': user_message})
    
    return response


def process_user_question(user_question):
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key=openai_api_key
    )
    
    # Loading the vector store
    vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    # Perform similarity search
    docs = vector_db.similarity_search(user_question)
    
    response = answer_code_question(user_question, docs)
    
    return response


def generate_code_explanation(python_code, python_name):
    
    prompt_template = """
    You are a Expert in Analysing and Explaining python code. Your task is to Explain the code given by the user line by line.
    Instructions:
        1. Initially state the name of the python file given to you for analysing.
        2. Explain the Logic of the code, Use the name of python file while explaining.
        3. Explain the whole code line by line, Use the name of python file while explaining.
    
    The name of Python file given : {file_name}
    
    The Python Code you need to explain : 
    {code}
    
    Answer : 
    """
    
    prompt = ChatPromptTemplate([('user', prompt_template)])
    
    chain = prompt | llm | parser
    
    response = chain.invoke({'file_name': python_name, 'code': python_code})
    
    return response


def generate_code_summary(python_code):
   
    try:
        user_template = """
        You are an expert programmer with deep knowledge of Python code. Provide a concise but comprehensive summary of the provided Python code.

        Instructions:
            1. Explain the overall purpose and functionality of the code.
            2. Identify the main components, classes, and functions.
            3. Describe the key algorithms or patterns used.
            4. Highlight any notable libraries or dependencies.
            5. Keep the summary clear and informative.

        Python code to summarize: {code}
        
        Your summary:
        """
        
        prompt = ChatPromptTemplate([('user', user_template)])
        
        chain = prompt | llm | parser
        
        response = chain.invoke({'code': python_code})
        
        logger.info("Successfully retrieved code summary from LLM")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in retrieving summary from OpenAI LLM: {e}")
        raise 