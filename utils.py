import os
import logging
import tempfile
import subprocess
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

openai_api_key= os.getenv("OPENAI_API_KEY")


def get_text_chunks(text):
    """Split text into manageable chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Store text chunks in Chroma vector database."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    
    # Make sure the directory exists
    if not os.path.exists("chroma_db"):
        os.makedirs("chroma_db")
        
    # Creating the vector store
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="chroma_db")
    
    logger.info("Successfully stored embeddings in Chroma DB")


def clone_repository(repo_url):
    """Clone a GitHub repository and extract Python files."""
    python_files = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_name = os.path.basename(repo_url)
        subprocess.run(["git", "clone", repo_url, tmpdir], check=True)
        logger.info(f"Successfully cloned Git Repo {repo_name}")

        # Retrieving the Python files
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        code = f.read()
                        python_files.append((file, code))
    
    return python_files 