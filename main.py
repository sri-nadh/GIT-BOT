import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models import RepoLink, UserMessage
import llm_operations as llm_ops
from utils import get_text_chunks, get_vector_store, clone_repository

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

openai_api_key = os.getenv("OPENAI_API_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as f:
        return f.read()


@app.post("/Git-bot/")
async def git_bot(user_query: UserMessage):
    
    logger.info("Entered Git-bot endpoint")
    
    try:
        # Process the user's question using the LLM operations
        llm_response = llm_ops.process_user_question(user_query.message)
        return llm_response
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process your question: {str(e)}")


@app.post("/summarize-code/")
async def analyze_repo(repo_link: RepoLink):
    """Analyze a GitHub repository, summarize the code and prepare for Q&A."""
    logger.info("Entered summarize-code endpoint")
    
    try:
        # Clone repository and extract Python files
        python_files = clone_repository(repo_link.repo_url)
        
        if not python_files:
            logger.warning("No Python files found in the repository.")
            return {".py": "NOT FOUND"}
            
        results = {}
        total_summary = []
        
        for filename, code in python_files:
            # Generate summary and detailed analysis
            summary = llm_ops.generate_code_summary(code)
            analysis = llm_ops.generate_code_explanation(code, filename)
            
            # Store results
            total_summary.append(analysis)
            results[filename] = summary
        
        # Prepare for vector search
        total_text = '\n\n'.join(total_summary)
        logger.info("Successfully appended all the Python file summaries")
        
        # Convert to text chunks and store in vector DB
        text_chunks = get_text_chunks(total_text)
        logger.info("Successfully converted text to chunks")
        
        get_vector_store(text_chunks)
        logger.info("Successfully stored text embeddings in database")
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing repository: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze repository: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 