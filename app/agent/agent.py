import os
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from pymongo import MongoClient
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv


class Symptom(TypedDict):
    name: str
    intensity: Optional[str]
    details: Optional[str]
    duration: Optional[str]
    frequency: Optional[str]
class AgentState(TypedDict):
    phone_number: str
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    
    name: Optional[str]
    age: Optional[int]
    main_complaint: Optional[str]
    symptoms_list: List[Symptom]
    symptom_to_process: List[str]
    history: Optional[str]
    measures_taken: Optional[str]

    triage_summary: Optional[str]

load_dotenv()

from app.agent.tools.prompts import EMERGENCY_KEYWORDS
from .tools import nodes

def triage_router(state: AgentState) -> str:
    user_message = state["messages"][-1]
    if any(keyword in user_message.content.lower() for keyword in EMERGENCY_KEYWORDS):
        print("--- ROTA: Detectada Emergência ---")
        return "emergency"

    if state.get("symptoms_to_process") and len(state["symptoms_to_process"]) > 0:
        print("--- ROTA: Ainda há sintomas para detalhar. ---")
        return "symptom_details"

    required_after_symptoms = ["history", "measures_taken"]
    if not state.get("symptoms_to_process") and all(state.get(field) for field in required_after_symptoms):
        print("--- ROTA: Triagem Completa. Gerando resumo. ---")
        return "summarize"
    
    print("--- ROTA: Continuando triagem (perguntando sobre histórico, etc.). ---")
    return "triage"

workflow = StateGraph(AgentState)

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
memory = MongoDBSaver(client, "conversations", state_id_key="phone_number")

workflow.add_node("triage", nodes.triage_node)
workflow.add_node("extract_data", nodes.extract_data_node)
workflow.add_node("symptom_details", nodes.symptom_details_node)
workflow.add_node("summarize", nodes.summarize_node)
workflow.add_node("emergency", nodes.emergency_node)

workflow.set_entry_point("triage")

workflow.add_edge("triage", "extract_data")
workflow.add_edge("symptom_details", "extract_data")
workflow.add_conditional_edges(
    "extract_data",
    triage_router,
    {
        "triage": END,
        "summarize": "summarize",
        "symptom_details": "symptom_details",
        "emergency": "emergency"
    }
)

workflow.add_edge("summarize", END)
workflow.add_edge("emergency", END)

triage_agent = workflow.compile(checkpointer = memory)

