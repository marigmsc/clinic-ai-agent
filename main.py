import os
import requests
import json
from flask import Flask,request,jsonify
from app.agent import triage_agent
from pymongo import MongoClient
from datetime import datetime

EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL", "http://localhost:8080")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "X9wxLzQe9RlRZhoSa1jREWgAFAZymJhMtGyR7klHt1k3Q54XlG4TkkP2v7Mus09b")
EVOLUTION_INSTANCE_NAME = os.getenv("EVOLUTION_INSTANCE_NAME", "default")

app = Flask(__name__)

def save_summary_to_mongodb(summary_data):
    mongodb_uri = os.getenv("MONGO_URI")
    if not mongodb_uri:
        print("Erro: MONGO_URI não encontrado para salvar o resumo.")
        return
    try:
        client = MongoClient(mongodb_uri)
        db = client["clinicai_db"]
        summaries_collection = db["summaries"]
        result = summaries_collection.insert_one(summary_data)
        print(f"✅ Resumo salvo com sucesso no MongoDB com o ID: {result.inserted_id}")
    except Exception as e:
        print(f"❌ Erro ao salvar o resumo no MongoDB: {e}")
    finally:
        if 'client' in locals() and client:
            client.close()


def send_whatsapp_message(phone_number, message_text):
    """
    Sends a text message to a given phone number using the Evolution API.
    """
    api_endpoint = f"{EVOLUTION_API_URL}/message/sendText/{EVOLUTION_INSTANCE_NAME}"
    headers = {
        'Content-Type': 'application/json',
        'apikey': EVOLUTION_API_KEY
    }
    payload = {
        "number": phone_number,
        "textMessage": {
            "text": message_text
        }
    }

    try:
        response = requests.post(api_endpoint, json=payload, headers=headers)
        response.raise_for_status()  
        print(f"Successfully sent message to {phone_number}. Response: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending message to {phone_number}: {e}")
        return None

@app.route('/webhook', methods=['POST'])
def whatsapp_webhook():
    """
    This endpoint receives webhook notifications from the Evolution API
    when a new message arrives.
    """
    webhook_data = request.json
    print("--- New Webhook Received ---")
    print(json.dumps(webhook_data, indent=2))

    if webhook_data.get('event') == 'messages.upsert' and 'data' in webhook_data:
        message_data = webhook_data['data']
        
        if not message_data.get('key', {}).get('fromMe', True):
            sender_number = message_data.get('key', {}).get('remoteJid')
            
            message_content = ""
            if 'message' in message_data:
                msg = message_data['message']
                if 'conversation' in msg:
                    message_content = msg['conversation']
                elif 'extendedTextMessage' in msg and 'text' in msg['extendedTextMessage']:
                    message_content = msg['extendedTextMessage']['text']

            if sender_number and message_content:
                config = {"configurable": {"thread_id": sender_number}}
                response = triage_agent.invoke(
                    {"messages": [("human", message_content)]},
                    config=config
                )
                if response.get("triage_summary"):
                    print(f"Resumo detectado para o usuário {sender_number}. Salvando...")
                    
                    summary_document = {
                        "phone_number": sender_number,
                        "patient_name": response.get("name"),
                        "patient_age": response.get("age"),
                        "main_complaint": response.get("main_complaint"),
                        "summary_text": response.get("triage_summary"),
                        "created_at": datetime.now() 
                    }
                    
                    save_summary_to_mongodb(summary_document)
                agent_reply = response['messages'][-1].content
                send_whatsapp_message(sender_number, agent_reply)

    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)