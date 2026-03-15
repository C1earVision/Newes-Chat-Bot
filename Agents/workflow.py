from Utils.ModelLoader import modelLoader
from langgraph.graph import StateGraph, END, START
from Tools.query_data_base import queryDataBase
from pydantic import BaseModel, Field
from typing import List, TypedDict, Dict


class QueryList(BaseModel):
    queries: List[str] = Field(description="List of focused search queries in Arabic")

class FinalResponse(BaseModel):
    response: str = Field(description="The final combined answer in Arabic")


class SubAnswerResponse(BaseModel):
    answer: str = Field(description="The summarized answer in Arabic based purely on the retrieved documents")

class GeneralAgent:
    def __init__(self, sysPrompt, modelName, output_schema=QueryList):
        self.model_loader = modelLoader(modelName)
        self.llm = self.model_loader.load_llm()
        self.system_prompt = sysPrompt
        self.structured_llm = self.llm.with_structured_output(output_schema)

    def invoke(self, user_message):
        messages = [self.system_prompt, ("human", user_message)]
        return self.structured_llm.invoke(messages)

class DataBaseQueryAgent(GeneralAgent):
    def __init__(self, sysPrompt, modelName):
        super().__init__(sysPrompt, modelName, SubAnswerResponse)

class FinalAnswerAgent(GeneralAgent):
    def __init__(self, sysPrompt, modelName):
        super().__init__(sysPrompt, modelName, FinalResponse)

class WorkflowState(TypedDict):
    original_query: str
    queries: List[str]
    answers: List[str]
    final_response: str

class ChatWorkflowBuilder:
    def __init__(self, general_prompt, db_prompt, final_prompt, models: Dict[str, str]):
        self.general_agent = GeneralAgent(sysPrompt=general_prompt, modelName=models["general"])
        self.db_agent = DataBaseQueryAgent(sysPrompt=db_prompt, modelName=models["db"])
        self.final_agent = FinalAnswerAgent(sysPrompt=final_prompt, modelName=models["final"])
        self.vector_db = queryDataBase()
        
    def generate_queries(self, state: WorkflowState) -> WorkflowState:
        print(f"Node: Generating Queries for: {state['original_query']}")
        result = self.general_agent.invoke(state["original_query"])
        return {"queries": result.queries}

    def retrieve_and_summarize(self, state: WorkflowState) -> WorkflowState:
        print(f"Node: Retrieving and Summarizing {len(state['queries'])} queries")
        answers = []
        for q in state["queries"]:
            db_results = self.vector_db.search(q)
            docs = []
            if db_results and 'documents' in db_results and db_results['documents']:
                for doc_group in db_results['documents']:
                    docs.extend(doc_group)
            
            combined_docs = "\n\n".join(docs)
            prompt_input = f"Search Query: {q}\n\nSearch Results:\n{combined_docs}"
            
            dbResponse = self.db_agent.invoke(prompt_input).answer
            answers.append(dbResponse)
        
        return {"answers": answers}
        
    def synthesize_final_answer(self, state: WorkflowState) -> WorkflowState:
        print(f"Node: Synthesizing Final Answer")
        finalInput = str({"original_question": state["original_query"], "answers": state["answers"]})
        final_output = self.final_agent.invoke(finalInput)
        return {"final_response": final_output.response}
        
    def build(self):
        builder = StateGraph(WorkflowState)
        
        builder.add_node("generate_queries", self.generate_queries)
        builder.add_node("retrieve_and_summarize", self.retrieve_and_summarize)
        builder.add_node("synthesize_final_answer", self.synthesize_final_answer)
        
        builder.add_edge(START, "generate_queries")
        builder.add_edge("generate_queries", "retrieve_and_summarize")
        builder.add_edge("retrieve_and_summarize", "synthesize_final_answer")
        builder.add_edge("synthesize_final_answer", END)
        
        return builder.compile()
