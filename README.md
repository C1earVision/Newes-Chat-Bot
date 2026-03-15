# Newes RAG ChatBot

A powerful Retrieval-Augmented Generation (RAG) ChatBot application built with FastAPI, LangGraph, and ChromaDB. This system scrapes a variety of designated websites to build a custom knowledge base, which its AI agents query to provide factual, up-to-date, and domain-specific answers.

## Features

- **Automated Web Scraping:** A robust, multithreaded web scraper built with `BeautifulSoup` and `requests`. It recursively crawls and extracts articles from several target websites (e.g., Ahram, Gomhuriaonline, Azhar, Dar-alifta, EgyptAir, etc.). Scraping runs periodically via an asynchronous background task.
- **Vector Database Storage:** Uses `SentenceTransformer` (`intfloat/multilingual-e5-small`) to convert text chunks into embeddings, storing them in a persistent `ChromaDB` collection (`website_articles`).
- **Agentic Workflow:** Employs `LangGraph` and LangChain to orchestrate a multi-agent system:
  - **General Agent:** Splits the users query into multible queries based on how many questions the user asked in the query and sends them to the database query agent.
  - **Database Query Agent:** Constructs search queries and retrieves top matches from ChromaDB.
  - **Final Answer Agent:** Synthesizes the retrieved context and user query into a coherent, accurate response.
- **FastAPI Backend:** Serves the frontend application (`static/index.html`) and provides the `/chat` endpoint for interacting with the AI.

## Project Structure

```
Kadash/ChatBot/
├── Agents/
│   └── workflow.py                  # LangGraph state graph for the multi-agent system
├── Data Processing Pipeline/
│   └── chroma_db/                   # Persistent vector database directory
│   └── scraped_data.json            # Cached, scraped articles
├── SysPrompts/                      # System prompts for different LLM agents
│   ├── general_agent_prompt.py
│   ├── query_data_base_agent_prompt.py
│   └── final_answer_agent_prompt.py
├── Tools/
│   ├── scraper.py                   # Async/multithreaded web scraping and ChromaDB integration logic
│   └── query_data_base.py           # Logic for embedding user queries and searching ChromaDB
├── Utils/                           # Helper utilities
├── static/                          # Frontend files served statically via FastAPI
├── .env                             # Environment variables defining agent model names and API keys
├── main.py                          # FastAPI server initialization, endpoints, and background scraper
├── requirements.txt                 # Project Python dependencies
└── README.md                        # Project documentation
```

## Requirements

The core dependencies are listed in `requirements.txt`:
* `fastapi`
* `uvicorn`
* `langchain`, `langchain-core`, `langchain-groq`, `langgraph`
* `chromadb`, `sentence-transformers`
* `requests`, `beautifulsoup4`
* `python-dotenv`

## Setup and Installation

1. **Clone the project & Navigate to the directory:**
   ```bash
   cd "d:/Files/Programing/AgenticAI/website question answer rag/Kadash/ChatBot"
   ```

2. **Set up a Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create your `.env` file to include required model configurations and any API keys:
   ```env
   # Example variables. Replace with actual model identifiers / keys.
   GENERAL_AGENT_MODEL_NAME=your_model_name
   DATA_BASE_QUERY_AGENT_MODEL_NAME=your_db_model_name
   FINAL_AGENT_MODEL_NAME=your_final_model_name
   GROQ_API_KEY=your_groq_api_key
   ```

## Running the Application

Start the backend server using Uvicorn:

```bash
uvicorn main:app --reload --port 8080
```

- When the server starts, an asynchronous background task detects whether the existing scraped data is older than 1 hour or absent. If so, it will automatically initiate a scrape and rebuild the ChromaDB embeddings.
- Open your browser and navigate to `http://localhost:8080/` to use the chatbot interface.
- Programmatic API requests can be sent via POST to `http://localhost:8080/chat` in the form `{"query": "your question"}`.
