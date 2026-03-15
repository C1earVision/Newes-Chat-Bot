from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from Agents.workflow import ChatWorkflowBuilder
from SysPrompts.general_agent_prompt import GENERAL_SYSTEM_PROMPT
from SysPrompts.query_data_base_agent_prompt import QUERY_DATA_BASE_AGENT_PROMPT
from SysPrompts.final_answer_agent_prompt import FINAL_ANSWER_AGENT_PROMPT
from starlette.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from contextlib import asynccontextmanager
from Tools.scraper import Scraper
import asyncio
import os
load_dotenv()


def _run_scraper_sync():
    try:
        scraper = Scraper()
        scraper.scrape_and_store()
        print("Scraper completed successfully.")
    except Exception as e:
        print(f"Scraper error: {e}")

def _should_scrape():
    data_file = "Data Processing Pipeline/scraped_data.json"
    if not os.path.exists(data_file):
        return True
    import time
    age = time.time() - os.path.getmtime(data_file)
    return age > 3600

async def run_scraper():
    await asyncio.to_thread(_run_scraper_sync)

async def periodic_scraper():
    if _should_scrape():
        print("Running initial scrape...")
        await run_scraper()
    else:
        print("Skipping initial scrape - data is recent.")
    while True:
        await asyncio.sleep(3600)
        print("Running scheduled scrape...")
        await run_scraper()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting...")
    task = asyncio.create_task(periodic_scraper())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

class Query(BaseModel):
    query: str
    admin: bool = False

@app.post("/chat")
async def chat(query: Query):
    try:
        GENERAL_AGENT_MODEL_NAME = os.environ.get("GENERAL_AGENT_MODEL_NAME")
        DATA_BASE_QUERY_AGENT_MODEL_NAME = os.environ.get("DATA_BASE_QUERY_AGENT_MODEL_NAME")
        FINAL_AGENT_MODEL_NAME = os.environ.get("FINAL_AGENT_MODEL_NAME")

        models = {
            "general": GENERAL_AGENT_MODEL_NAME,
            "db": DATA_BASE_QUERY_AGENT_MODEL_NAME,
            "final": FINAL_AGENT_MODEL_NAME
        }

        workflow_builder = ChatWorkflowBuilder(
            general_prompt=GENERAL_SYSTEM_PROMPT,
            db_prompt=QUERY_DATA_BASE_AGENT_PROMPT,
            final_prompt=FINAL_ANSWER_AGENT_PROMPT,
            models=models
        )
        
        graph = workflow_builder.build()
        initial_state = {"original_query": query.query}
        
        print(f"Starting workflow for query: {query.query}")
        final_state = graph.invoke(initial_state)
        finalResponse = final_state["final_response"]
        
        print(f"Final Response: {finalResponse}")

        return JSONResponse(status_code=200, content={"response": finalResponse})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
