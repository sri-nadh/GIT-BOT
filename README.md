# GIT-BOT: GitHub Code Analysis Assistant

A powerful tool for analyzing GitHub repositories, generating code summaries, and answering questions about codebases using AI.

## Features

- **Repository Analysis**: Clone and analyze GitHub repositories
- **Code Summarization**: Generate concise, comprehensive summaries of Python files
- **Detailed Explanations**: Line-by-line analysis of code functionality
- **Interactive Q&A**: Ask questions about the codebase and receive detailed answers
- **Modern Web Interface**: Clean, responsive design with tabbed interface

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/GIT-BOT.git
   cd GIT-BOT
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. Start the server:
   ```
   python main.py
   ```

2. Open your browser and navigate to `http://localhost:8000`

3. In the Code Analyzer tab:
   - Enter a GitHub repository URL
   - Click "Analyze Repository"
   - View the generated summaries for each Python file

4. In the Assistant tab:
   - Ask questions about the analyzed codebase
   - Get AI-powered answers with context from the code

## Project Structure

- `main.py`: FastAPI application and API endpoints
- `models.py`: Pydantic data models
- `utils.py`: Helper functions for text processing and repository operations
- `llm_operations.py`: LLM-based code analysis and question answering
- `static/`: Frontend assets (HTML, CSS, JavaScript)

## How It Works

1. When you submit a repository URL, GIT-BOT clones the repository
2. It extracts all Python files and processes them with GPT-4
3. Each file receives both a summary and detailed explanation
4. The explanations are embedded into a vector database (ChromaDB)
5. When you ask questions, relevant code explanations are retrieved and used to provide accurate answers



