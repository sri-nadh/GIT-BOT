from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from pydantic import BaseModel
import logging
import os
import tempfile
import subprocess
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

openai_api_key=os.getenv("OPENAI_API_KEY")
google_api_key=os.getenv("GOOGLE_API_KEY")

logging.basicConfig(level=logging.INFO,)
logger=logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["GET","POST"],  
    allow_headers=["*"], 
)


app.mount("/static", StaticFiles(directory="static"), name="static")

#loading the html file from static dir
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as f:
        return f.read()


class RepoLink(BaseModel):
    repo_url: str
    
class userMessage(BaseModel):
    message: str


#Function which returns the response for the user queries with context taken from vector database(FAISS)
def llm_bot(user_message,code_context):

    model=ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.2)
    parser= StrOutputParser()
    prompt_template = """
    Answer the question as detailed as possible from the provided context(Python code Explanation) , make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer.
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    prompt= ChatPromptTemplate([('user',prompt_template)])
    
    chain= prompt | model | parser
    
    response= chain.invoke({'context':code_context,'question':user_message})
    
    return response

#function to convert normal text to chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#function to store vector embedding in faiss 
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    # Make sure the directory exists
    if not os.path.exists("chroma_db"):
        os.makedirs("chroma_db")
    # Create and return the vector store (no need to call persist with langchain_chroma)
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="chroma_db")
    logger.info("Successfully stored embeddings in Chroma DB")

#function handling the user queries about the code
def user_input(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    
    new_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    docs = new_db.similarity_search(user_question)
    response = llm_bot(user_question, docs)
    return response


#Endpoint for the bot
@app.post("/Git-bot/")
async def Git_Bot(user_query: userMessage):
    logger.info("Entered Git-bot endpoint")
    
    llm_response=user_input(user_query.message)
    
    return llm_response


#Function to analyse each python file , which is then embedded and stored in vector database
def Code_Detailed_analysis(python_code,python_name):

    model=ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.2)
    
    parser= StrOutputParser()
    
    prompt_template="""
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
    
    prompt= ChatPromptTemplate([('user',prompt_template)])
    
    chain= prompt | model | parser
    
    response= chain.invoke({'file_name':python_name,'code':python_code})
    
    return response



#function which summarizes Python code using LLM
def summarise_python_file(python_code):
    try:
        code = python_code
    
        model = ChatOpenAI(model="gpt-4.1")
    
        user_template="""
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
        parser = StrOutputParser()
    
        prompt = ChatPromptTemplate([('user', user_template)])

        chain = prompt | model | parser
    
        response = chain.invoke({'code': code})

        logger.info("Successfully retrieved code summary from LLM")
        return response
    
    except Exception as e:
        logger.error(f"Error in retrieving summary from OpenAI LLM: {e}")
        raise
        

@app.post("/summarize-code/")
async def analyze_repo(repo_link: RepoLink):
    logger.info("Entered summarize-code endpoint")
    
    python_files = []
    total_summary=[]
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_name = os.path.basename(repo_link.repo_url)
            subprocess.run(["git", "clone", repo_link.repo_url, tmpdir], check=True)
            logger.info(f"Successfully cloned Git Repo {repo_name}")
    
            #Retrieving the Python files
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r") as f:
                            code = f.read()
                            python_files.append((file, code))
        
            if not python_files:
                logger.warning("No Python files found in the repository.")
                return {".py":"NOT FOUND"}
                
    except Exception as e:
        logger.error(f"Error in cloning the repo or retrieving Python files: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process the repository.")

    #code analysis with llm
    try:
        results = {}
        total_text=''
        
        for filename, code in python_files:
            summary = summarise_python_file(code)
            analysis = Code_Detailed_analysis(code,filename)
            total_summary.append(analysis)
            
            # Store the summary directly
            results[filename] = summary
        
        
        for file in total_summary:
            total_text+=file +'\n\n'
        logger.info(f"Successfully appended all the Python file summary in a single string")
        
        text_chunks= get_text_chunks(total_text)
        logger.info(f"Successfully converted normal text to text chunks ")
        
        get_vector_store(text_chunks)
        logger.info(f"Successfully stored text embedding in FAISS ")
        
        
        logger.info(f"Successfully analyzed all the Python files in the Git repo, result: {results}")
        
        return results

    except Exception as e:
        logger.error(f"Error in analyzing the Python files or Error in embedding the text for the bot: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze the Python files.")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





