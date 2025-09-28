import os
import logging
# from prompts import SYMPTOM_DETAILS_PROMPT, EXTRACT_PROMPT, PERSONA_PROMPT, EMERGENCY_KEYWORDS, SUMMARY_PROMPT
import json 
from typing import TypedDict, Annotated, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
logging.getLogger('grpc._plugin_wrapping').setLevel(logging.ERROR)

SYMPTOM_DETAILS_PROMPT = """
Sua missão AGORA é focar em investigar UM ÚNICO sintoma do paciente.
O paciente já listou os sintomas. Agora, você precisa detalhar o seguinte sintoma: '{symptom_name}'

Faça perguntas para entender melhor este sintoma específico. Pergunte sobre:
- A intensidade (numa escala de 0 a 10).
- As características (é uma dor que queima, pulsa, é uma pontada?).
- Qualquer outra coisa que ajude a detalhar APENAS o sintoma '{symptom_name}'.

Histórico da Conversa:
{messages}

Faça a próxima pergunta focada neste sintoma.
"""
PERSONA_PROMPT = """

**Sua Primeira Ação:**
Ao receber a primeira mensagem do usuário (como "oi" ou "olá"), você DEVE se apresentar e fazer a primeira pergunta. Responda com: "Olá! Sou um assistente virtual da ClinicAI. Meu objetivo é coletar algumas informações para agilizar sua consulta. Lembre-se, não substituo a avaliação de um profissional de saúde. Para começar, poderia me informar seu nome completo e sua idade, por favor?"
Se na resposta dessa pergunta houver vários dados(sintomas), pergunte qual o sintoma que mais está incomodando o paciente. Após isso peça detalhadamente todos os sintomas. Pergunte da frequência, intensidade e duração de cada sintoma individualmente e só depois deve seguir para Medidas tomadas e Histórico. 
**Sua Persona:**
- Acolhedor e Empático: Use uma linguagem que acalme o paciente.
- Calmo e Profissional: Mantenha um tom profissional.
- Linguagem Simples: Evite jargões médicos.

**Sua Missão - Coletar os Seguintes Pontos (UM DE CADA VEZ):**
1. Nome e Idade
2. Queixa Principal
3. Sintomas Detalhados
4. Duração 
5. Frequência
6. Intensidade (escala 0 a 10)
7. Histórico Relevante/ Medicamentos regulares
8. Medidas Tomadas/ Medicamentos tomadas para a queixa

**Regras CRÍTICAS de Segurança:**
- Evite fazer mais de uma pergunta de uma vez se elas possuem respostas diferentes
- NUNCA ofereça diagnósticos.
- NUNCA sugira tratamentos ou medicamentos.
- PROTOCOLO DE EMERGÊNCIA: Se o usuário mencionar palavras-chave de emergência (dor no peito, falta de ar, desmaio, etc.), interrompa a triagem e responda APENAS com a mensagem de emergência padrão.
"""
EXTRACT_PROMPT = """
Sua tarefa é analisar uma conversa de triagem e extrair/atualizar informações em um formato JSON.
Se na resposta de principal queixa houver mais de uma sintoma, deve ser armazenado em sintomas. Quando o usuário responder qual a maior queixa dele entre esses sintomas(até 2) você guarda no main_complaint
**Contexto Atual:**
- Sintomas na fila para detalhar: {symptoms_to_process}
- Sintomas já detalhados: {symptoms_list}

**Sua Missão:**
1.  **Se o usuário acabou de listar os sintomas pela primeira vez:** Extraia os NOMES dos sintomas e coloque-os no campo `new_symptoms_to_process`.
2.  **Se o usuário está respondendo a uma pergunta sobre um sintoma específico (o primeiro da fila `symptoms_to_process`):** Extraia os detalhes (intensidade, duração, frequência, etc.) e preencha o campo `symptom_update`. Certifique-se de incluir o nome do sintoma que está sendo atualizado.
3.  **Para outros dados (nome, idade, histórico, medidas tomadas):** Preencha os campos correspondentes.
4.  Mantenha os dados já extraídos anteriormente no estado. Sua saída deve ser APENAS o JSON.

**Histórico da Conversa:**
{messages}

**Estado Atual (para referência):**
{state}

**Preencha o seguinte JSON:**
{{
    "name": "string or null",
    "age": "integer or null",
    "main_complaint": "string or null",
    "history": "string or null",
    "measures_taken": "string or null",
    "new_symptoms_to_process": ["sintoma1", "sintoma2", ...],
    "symptom_update": {{
        "name": "nome do sintoma sendo atualizado",
        "intensity": "string or null",
        "details": "string or null",
        "duration": "string or null",
        "frequency": "string or null"
    }}
}}
"""
SUMMARY_PROMPT = """
Você é um assistente de IA encarregado de resumir as informações de triagem de um paciente para uma equipe médica.
Com base nos dados coletados abaixo, crie um resumo claro, objetivo e estruturado. Use um formato de tópicos (bullet points).
**DADOS DO PACIENTE:**
- Nome: {name}
- Idade: {age}
Dados Coletados:
- Queixa Principal: {main_complaint}
- Histórico Relevante: {history}
- Medidas Já Tomadas: {measures_taken}
**DETALHAMENTO DOS SINTOMAS:**
{symptoms_summary}

Gere o resumo para ser anexado ao prontuário do paciente.
"""
EMERGENCY_KEYWORDS = [
    "dor no peito", "aperto no peito", "falta de ar", "desmaio", 
    "sangramento intenso", "dificuldade de falar", "perda de força",
    "infarto", "avc", "embolia"
]

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
    # duration: Optional[str]
    # frequency: Optional[str]
    # intensity: Optional[str]
    history: Optional[str]
    measures_taken: Optional[str]

    triage_summary: Optional[str]

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

# ---------------------------------------------- Nodes -------------------------------------------------------------

# def start_node(state: AgentState) -> AgentState:

#     welcome_message = AIMessage(
#         content=(
#             "Olá! Sou um assistente virtual da ClinicAI. Meu objetivo é coletar algumas informações "
#             "para agilizar sua consulta. Lembre-se, não substituo a avaliação de um profissional de saúde.\n\n"
#             "Para começar, por favor, me diga qual é o motivo principal do seu contato?"
#         )
#     )
#     return {"messages": [welcome_message]}
def symptom_details_node(state: AgentState) -> AgentState:

    print(f"--- NÓ: Detalhando Sintoma ---")
    
    symptom_to_ask_about = state["symptoms_to_process"][0]
    
    prompt = ChatPromptTemplate.from_template(SYMPTOM_DETAILS_PROMPT)
    
    chain = prompt | llm
    response = chain.invoke({
        "symptom_name": symptom_to_ask_about,
        "messages": state["messages"]
    })
    
    return {"messages": [response]}
def summarize_node(state: AgentState) -> AgentState:
    print("--- NÓ: Gerando Resumo ---")
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

    summary_prompt_formatted = SUMMARY_PROMPT.format(
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
    system_message = SystemMessage(content=PERSONA_PROMPT)
    
    messages_for_llm = [system_message] + state["messages"]
    
    response = llm.invoke(messages_for_llm)
    
    return {"messages": [response]}

def emergency_node(state: AgentState) -> AgentState:
    """
    Nó de emergência: envia a mensagem padrão de emergência.
    """
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

    prompt = ChatPromptTemplate.from_template(EXTRACT_PROMPT)
    
    extractor_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
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


# ------------------------------------------------ Routers -------------------------------------------------

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

# ------------------------------------------- Graph ----------------------------------------------------------------

workflow = StateGraph(AgentState)
memory = MemorySaver()

# workflow.add_node("start", start_node)
workflow.add_node("triage", triage_node)
workflow.add_node("extract_data", extract_data_node)
workflow.add_node("symptom_details", symptom_details_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("emergency", emergency_node)

workflow.set_entry_point("triage")
workflow.add_edge("triage", "extract_data")
workflow.add_edge("symptom_details", "extract_data")

# workflow.add_conditional_edges(
#     "start",
#     triage_router,
#     {
#         "triage": "triage",
#         "emergency": "emergency"
#     }
# )
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

triage_agent = workflow.compile(checkpointer =memory)

# if __name__ == '__main__':
#     config = {"configurable": {"thread_id": "test-thread-1"}}
    
#     print("Iniciando a conversa... Diga 'oi' para começar.")

#     while True:
#         user_input = input("Você: ")
#         if user_input.lower() in ["sair", "exit"]:
#             break
            
#         response = triage_agent.invoke(
#             {"messages": [HumanMessage(content=user_input)]}, 
#             config
#         )
        
#     
#         current_state = triage_agent.get_state(config)
#         print("\n--- ESTADO ATUAL ---")
#         for key, value in current_state.values.items():
#             if key != 'messages':
#                 print(f"{key}: {value}")
#         print("---------------------\n")
        
#         print("Agente:", response['messages'][-1].content)
        
#         if "procure o pronto-socorro" in response['messages'][-1].content or "Sua triagem inicial foi concluída" in response['messages'][-1].content :
#             break