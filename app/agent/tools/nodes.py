import json 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from . import prompts 
from ..agent import AgentState, Symptom

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
extractor_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def symptom_details_node(state: AgentState) -> AgentState:

    print(f"--- NÓ: Detalhando Sintoma ---")
    
    symptom_to_ask_about = state["symptoms_to_process"][0]
    
    prompt = ChatPromptTemplate.from_template(prompts.SYMPTOM_DETAILS_PROMPT)
    
    chain = prompt | llm
    response = chain.invoke({
        "symptom_name": symptom_to_ask_about,
        "messages": state["messages"]
    })
    
    return {"messages": [response]}

def summarize_node(state: AgentState) -> AgentState:
    print("--- NÓ: Gerando Resumo ---")

    symptoms_summary = ""
    for s in state.get("symptoms_list", []):
        symptoms_summary += f"\n- Sintoma: {s.get('name')}\n"
        if s.get('intensity'):
            symptoms_summary += f"  - Intensidade: {s.get('intensity')}\n"
        if s.get('duration'):
            symptoms_summary += f"  - Duração: {s.get('duration')}\n"
        if s.get('frequency'):
            symptoms_summary += f"  - Frequência: {s.get('frequency')}\n"
        if s.get('details'):
            symptoms_summary += f"  - Detalhes: {s.get('details')}\n"

    summary_prompt_formatted = prompts.SUMMARY_PROMPT.format(
        name=state.get("name"),
        age=state.get("age"),
        main_complaint=state.get("main_complaint"),
        symptoms_summary=symptoms_summary, 
        history=state.get("history"),
        measures_taken=state.get("measures_taken")
    )

    summary_response = llm.invoke(summary_prompt_formatted)
    summary_text = summary_response.content

    final_message = AIMessage(
        content=(
            "Obrigado por compartilhar as informações. Sua triagem inicial foi concluída. "
            "Uma cópia das suas respostas foi enviada para nossa equipe, que entrará em contato em breve para agendar sua consulta. "
            "Seus dados estão seguros conosco."
        )
    )

    return {
        "triage_summary": summary_text,
        "messages": [final_message]
    }

def triage_node(state: AgentState) -> AgentState:
    system_message = SystemMessage(content=prompts.PERSONA_PROMPT)
    
    messages_for_llm = [system_message] + state["messages"]
    
    response = llm.invoke(messages_for_llm)
    
    return {"messages": [response]}

def emergency_node(state: AgentState) -> AgentState:
    print("--- NÓ: Situação de Emergência Detectada ---")
    emergency_message = AIMessage(
        content=(
            "Seus sintomas podem indicar uma situação de emergência. Por favor, "
            "procure o pronto-socorro mais próximo ou ligue para o 192 (SAMU) imediatamente."
        )
    )
    return {"messages": [emergency_message]}

def extract_data_node(state: AgentState) -> AgentState:
    
    if "symptoms_list" not in state or state["symptoms_list"] is None:
        state["symptoms_list"] = []
    if "symptoms_to_process" not in state or state["symptoms_to_process"] is None:
        state["symptoms_to_process"] = []

    prompt = ChatPromptTemplate.from_template(prompts.EXTRACT_PROMPT)
    
    chain = prompt | extractor_llm
    
    response = chain.invoke({
        "messages": state["messages"],
        "symptoms_to_process": state["symptoms_to_process"],
        "symptoms_list": state["symptoms_list"],
        "state": {k: v for k, v in state.items() if k != "messages"}
    })

    try:
        json_str = response.content.strip().replace("```json", "").replace("```", "").strip()
        extracted_data = json.loads(json_str)

        updates = {}

        for key in ["name", "age", "main_complaint", "history", "measures_taken"]:
            if extracted_data.get(key) is not None:
                updates[key] = extracted_data[key]
        
        new_symptoms_names = extracted_data.get("new_symptoms_to_process", [])
        if new_symptoms_names:
            current_symptom_names = {s['name'].lower() for s in state["symptoms_list"]}
            truly_new_symptoms = [s_name for s_name in new_symptoms_names if s_name.lower() not in current_symptom_names]
            
            if truly_new_symptoms:
                updates["symptoms_to_process"] = state["symptoms_to_process"] + truly_new_symptoms
                new_symptoms_obj = [Symptom(name=s) for s in truly_new_symptoms]
                updates["symptoms_list"] = state["symptoms_list"] + new_symptoms_obj

        if extracted_data.get("symptom_update") and extracted_data["symptom_update"].get("name"):
            update_details = extracted_data["symptom_update"]
            symptom_name_to_update = update_details["name"]
            
            current_list = state["symptoms_list"]
            
            for symptom in current_list:
                if symptom["name"].lower() == symptom_name_to_update.lower():
                    if update_details.get("intensity") is not None:
                        symptom["intensity"] = update_details["intensity"]
                    if update_details.get("details") is not None:
                        symptom["details"] = update_details["details"]
                    if update_details.get("duration") is not None:
                        symptom["duration"] = update_details["duration"]
                    if update_details.get("frequency") is not None:
                        symptom["frequency"] = update_details["frequency"]
                    break 
            
            updates["symptoms_list"] = current_list
            
            current_queue = state["symptoms_to_process"]
            if current_queue and current_queue[0].lower() == symptom_name_to_update.lower():
                if all(symptom.get(key) for key in ["intensity", "duration", "frequency"]):
                    updates["symptoms_to_process"] = current_queue[1:]


        return updates

    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Erro ao decodificar JSON ou processar dados: {e}")
        return {}