from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
import logging
import os
import tempfile
import subprocess
import json

app = FastAPI()


api_key=''

logging.basicConfig(level=logging.INFO,)
logger=logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
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
    model=ChatGoogleGenerativeAI(model='gemini-1.5-flash',api_key=api_key)
    parser= StrOutputParser()
    prompt_template = """
    Answer the question as detailed as possible from the provided context(Python code Explanation) , make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

#function handling the user queries about the code
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    response = llm_bot(user_question,docs)
    return response


#Endpoint for the bot
@app.post("/Git-bot/")
async def analyze_repo(user_query: userMessage):
    logger.info("Entered Git-bot endpoint")
    
    llm_response=user_input(user_query.message)
    
    return llm_response


#Function to analyse each python file , which is then embedded and stored in vector database
def llm_analyse_chain(python_code,python_name):

    model=ChatGoogleGenerativeAI(model='gemini-1.5-flash',api_key=api_key)
    
    parser= StrOutputParser()
    
    prompt_template="""
    You are a Expert in Analysing and Explaining python code. Your task is to Explain the code given by the user line by line.
    Instructions:
    1. Initially state the name of the python file given to you for analysing.
    2. Explain the Logic of the code, Use the name of python file while explaining.
    3. Explain the whole code line by line, Use the name of python file while explaining.\n\n
    
    The name of Python file given : {file_name}\n
    The Python Code you need to explain : \n{code}
    
    Answer : 
    """
    
    prompt= ChatPromptTemplate([('user',prompt_template)])
    
    chain= prompt | model | parser
    
    response= chain.invoke({'file_name':python_name,'code':python_code})
    
    return response



#function which counts the no of if,for,while constructs in a python code using llm
def llm_count_chain(python_code):
    try :
        code= python_code
    
        #os.environ["OPENAI_API_KEY"]=''
        model = ChatOpenAI(model="gpt-4")
    
        user_template="""
You are an expert programmer with deep knowledge of Python code. Analyze the provided code to count the exact occurrences of if, for, and while constructs. 
Only count these keywords when they are used as actual constructs in the code, not when they appear inside strings or comments.

Instructions:

1. Analyse the code line by line.
2. Count the if, for, and while constructs.
3. Dont count 'if' construct twice when associated with the 'else' construct
4. Exclude any occurrences within strings or comments.
5. Identify and count nested constructs:
      a. if inside for (if_in_for)
      b. while inside if (while_in_if)
      c. for inside while (for_in_while)
      d. if inside while (if_in_while)
      e. for inside if (for_in_if)
      f. while inside for (while_in_for)

Return the results only in the following JSON format(dont include explanations):

  "if": <count>,
  "for": <count>,
  "while": <count>,
  "if_in_for": <count>,
  "for_in_if": <count>,
  "while_in_if": <count>,
  "if_in_while": <count>,
  "for_in_while": <count>,
  "while_in_for": <count> \n\n


Python code for which you need to count the contructs : {code}
 
your Answer:
"""
        parser=StrOutputParser()
    
        prompt=ChatPromptTemplate([('user',user_template)])

        chain = prompt | model | parser
    
        response=chain.invoke({'code':code})

        logger.info("Successfully retreived response from LLM")
        return response
    
    except Exception as e:
        logger.error(f"Error in retreiving response from Openai LLM{e}")
        raise
        

@app.post("/construct-count/")
async def analyze_repo(repo_link: RepoLink):
    logger.info("Entered construct-count endpoint")
    
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
            response = llm_count_chain(code)
            summary=llm_analyse_chain(code,filename)
            total_summary.append(summary)
            
            
            decoded_response = json.loads(response)
            
            results[filename] = decoded_response
        
        
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





