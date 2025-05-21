# GitHub Code Assistant

A modern web application that analyzes GitHub repositories, generates comprehensive code summaries, and provides an AI assistant to answer questions about the code.

## Features

- **Code Analysis**: Clone and analyze any public GitHub repository
- **Automatic Summarization**: Generate concise summaries of Python code files
- **Detailed Explanations**: Get line-by-line explanations of Python code
- **Interactive Q&A**: Ask questions about the analyzed code and get detailed answers
- **Modern UI**: Clean and responsive interface with tab-based navigation
- **Vector Search**: Efficient code information retrieval using embeddings

## Installation

### Prerequisites

- Python 3.8+
- Git
- OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sri-nadh/GIT-BOT.git
   cd GIT-BOT
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your browser and go to `http://localhost:8000`

3. In the "Code Analyzer" tab:
   - Enter a GitHub repository URL (e.g., https://github.com/username/repo)
   - Click "Analyze Repository"
   - View generated summaries for each Python file

4. In the "Code Assistant" tab:
   - Ask questions about the analyzed code
   - Get detailed answers based on the code context

## How It Works

1. **Repository Analysis**:
   - The application clones the specified GitHub repository
   - Identifies all Python files in the repository
   - Processes each file for analysis

2. **Code Understanding**:
   - Uses OpenAI's models to generate summaries and detailed explanations
   - Creates comprehensive code documentation automatically

3. **Vector Database**:
   - Stores code explanations in a Chroma vector database
   - Enables semantic search to find relevant code sections

4. **Interactive Q&A**:
   - Uses retrieval-augmented generation to answer specific questions
   - Provides contextually relevant explanations from the codebase

