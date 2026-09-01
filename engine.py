import datetime
import json
import os
import subprocess
import threading
import time
import requests
import chromadb
from ddgs import DDGS
import ollama
import psutil
import schedule
from pypdf import PdfReader

ASSISTANT_NAME = "BAT"

# ---------------------------------------------------------
# Live API & System Tools
# ---------------------------------------------------------
def get_system_stats() -> str:
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    battery = psutil.sensors_battery()
    bat_str = f"{battery.percent}%" if battery else "Desktop/Plugged In"
    return f"CPU: {cpu}% | RAM Usage: {ram}% | Battery: {bat_str}"

def open_application(app_name: str) -> str:
    apps = {"chrome": "start chrome", "notepad": "notepad", "cmd": "start cmd", "calculator": "calc", "explorer": "explorer"}
    target = app_name.lower().strip()
    if target in apps:
        subprocess.Popen(apps[target], shell=True)
        return f"Successfully opened {target}."
    else:
        try:
            subprocess.Popen(f"start {target}", shell=True)
            return f"Attempted to launch {target}."
        except Exception as e:
            return f"Failed to launch application: {e}"

def get_time_and_date() -> str:
    now = datetime.datetime.now()
    return f"Current System Date and Time: {now.strftime('%A, %B %d, %Y - %I:%M %p')}"

def get_weather(location: str) -> str:
    if not location:
        location = "Tirupati" 
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geo_data = requests.get(geo_url).json()
        if "results" not in geo_data:
            return f"Could not find location data for {location}."
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_data = requests.get(weather_url).json()
        temp = weather_data["current_weather"]["temperature"]
        wind = weather_data["current_weather"]["windspeed"]
        return f"Current weather in {location}: {temp}°C, Wind Speed: {wind} km/h."
    except Exception as e:
        return f"Weather API Error: {e}"

def get_f1_standings() -> str:
    try:
        url = "http://api.jolpi.ca/ergast/f1/current/driverStandings.json"
        data = requests.get(url).json()
        standings = data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings'][:5]
        result = "Top 5 F1 Drivers Current Standings:\n"
        for driver in standings:
            name = f"{driver['Driver']['givenName']} {driver['Driver']['familyName']}"
            points = driver['points']
            result += f"{driver['position']}. {name} ({points} pts)\n"
        return result.strip()
    except Exception as e:
        return f"F1 API Error: {e}"

def get_latest_news(topic: str) -> str:
    try:
        ddgs = DDGS()
        search_topic = topic if topic else "technology"
        results = ddgs.news(search_topic, max_results=3)
        return "\n".join([f"Headline: {r['title']} (Source: {r['source']})" for r in results])
    except Exception as e:
        return f"News API Error: {e}"

# ---------------------------------------------------------
# Self-Learning Brain Architecture
# ---------------------------------------------------------
class SelfLearningBrain:
    def __init__(self, model_name="llama3.2"):
        self.model = model_name
        self.chroma = chromadb.PersistentClient(path="D:/BatAI/brain_memory")
        self.knowledge_memory = self.chroma.get_or_create_collection(name="knowledge")
        self.user_memory = self.chroma.get_or_create_collection(name="user_profile")
        self.doc_memory = self.chroma.get_or_create_collection(name="local_documents")
        self.conversation_history = []

    def analyze_and_route(self, raw_input: str) -> dict:
        prompt = f"""
Analyze the user input. Correct all typos and informal slang.
Determine the primary intent and output ONLY raw JSON.

Available Intents:
- 'SYSTEM_STATS': Ask for CPU, RAM, or battery.
- 'OPEN_APP': Open a program. (Payload: app name)
- 'LEARN_WEB': Research a topic to save to memory.
- 'STORE_FACT': Remember a personal fact.
- 'GET_WEATHER': Ask for the weather. (Payload: city name, or empty for local)
- 'GET_TIME': Ask for current time, date, or calendar.
- 'GET_NEWS': Ask for latest news. (Payload: news topic)
- 'GET_F1_STANDINGS': Ask for F1 racing standings.
- 'READ_DOC': User wants to ingest a local document. (Payload: file path)
- 'CHAT': General question or querying stored documents/memory.

JSON Output Format:
{{
  "corrected_text": "<cleaned string>",
  "intent": "<ONE OF THE INTENTS ABOVE>",
  "payload": "<extracted target like city, file path, app, or topic>",
  "search_query": "<short query for memory>"
}}

User Input: "{raw_input}"
"""
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}], format="json")
        content = response['message']['content'].strip()
        
        try:
            parsed = json.loads(content)
            print(f"\n[DEBUG - Routing]: {parsed['intent']} | Payload: {parsed.get('payload', 'None')}")
            return parsed
        except Exception:
            return {"corrected_text": raw_input, "intent": "CHAT", "payload": "", "search_query": raw_input}

    def search_web(self, query: str) -> str:
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=3)
            return "\n".join([f"Source: {r['title']}\n{r['body']}" for r in results])
        except Exception as e:
            return f"Search error: {e}"

    def learn_topic(self, topic: str):
        content = self.search_web(topic)
        if content and "Search error" not in content:
            doc_id = f"{topic}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.knowledge_memory.add(documents=[content], metadatas=[{"topic": topic}], ids=[doc_id])
            print(f"[{ASSISTANT_NAME}] Stored '{topic}' to memory.")

    def save_user_fact(self, fact: str):
        doc_id = f"fact_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.user_memory.add(documents=[fact], metadatas=[{"type": "user_preference"}], ids=[doc_id])
        print(f"[{ASSISTANT_NAME}] Saved fact: '{fact}'")

    def read_document(self, file_path: str) -> str:
        """Reads a local file and stores chunks in ChromaDB."""
        clean_path = file_path.strip().strip('"').strip("'")
        if not os.path.exists(clean_path):
            return f"Error: Cannot find file at {clean_path}."
        
        text_content = ""
        try:
            if clean_path.lower().endswith(".pdf"):
                reader = PdfReader(clean_path)
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
            else:
                with open(clean_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
                    
            if not text_content.strip():
                return f"Error: The document {clean_path} appears to be empty or unreadable."

            # Chunk the text so the AI can digest it
            chunks = [text_content[i:i+1000] for i in range(0, len(text_content), 1000)]
            doc_ids = [f"doc_{os.path.basename(clean_path)}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": clean_path} for _ in chunks]
            
            self.doc_memory.add(documents=chunks, metadatas=metadatas, ids=doc_ids)
            return f"Successfully read and indexed {len(chunks)} chunks from {os.path.basename(clean_path)}."
        except Exception as e:
            return f"Failed to read document: {e}"

    def process(self, raw_input: str) -> str:
        parsed = self.analyze_and_route(raw_input)
        intent = parsed.get("intent", "CHAT")
        payload = parsed.get("payload", "")
        tool_output = ""

        # Routing Engine
        if intent == "SYSTEM_STATS": tool_output = get_system_stats()
        elif intent == "OPEN_APP" and payload: tool_output = open_application(payload)
        elif intent == "GET_TIME": tool_output = get_time_and_date()
        elif intent == "GET_WEATHER": tool_output = get_weather(payload)
        elif intent == "GET_NEWS": tool_output = get_latest_news(payload)
        elif intent == "GET_F1_STANDINGS": tool_output = get_f1_standings()
        elif intent == "LEARN_WEB" and payload: self.learn_topic(payload)
        elif intent == "STORE_FACT" and payload: self.save_user_fact(payload)
        elif intent == "READ_DOC" and payload: tool_output = self.read_document(payload)

        # Context retrieval across all memory banks
        search_q = parsed.get("search_query", raw_input)
        k_res = self.knowledge_memory.query(query_texts=[search_q], n_results=1)
        u_res = self.user_memory.query(query_texts=[search_q], n_results=1)
        d_res = self.doc_memory.query(query_texts=[search_q], n_results=2)
        
        context = ""
        if u_res.get("documents") and u_res["documents"][0]: 
            context += "USER PROFILE MEMORY:\n" + "\n".join(u_res["documents"][0]) + "\n\n"
        if k_res.get("documents") and k_res["documents"][0]: 
            context += "STORED KNOWLEDGE:\n" + "\n".join(k_res["documents"][0]) + "\n\n"
        if d_res.get("documents") and d_res["documents"][0]: 
            context += "DOCUMENT EXCERPTS:\n" + "\n".join(d_res["documents"][0]) + "\n\n"

        system_prompt = (
            f"You are {ASSISTANT_NAME}, an intelligent personal AI assistant. "
            "Formulate a direct, conversational response using the API/Tool results and context provided below.\n\n"
            f"API/TOOL RESULT:\n{tool_output}\n\n"
            f"RECALLED MEMORY:\n{context}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history[-6:])
        messages.append({"role": "user", "content": parsed.get("corrected_text", raw_input)})

        response = ollama.chat(model=self.model, messages=messages)
        answer = response["message"]["content"]

        self.conversation_history.append({"role": "user", "content": parsed.get("corrected_text", raw_input)})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return answer

if __name__ == "__main__":
    brain = SelfLearningBrain()
    print(f"{ASSISTANT_NAME} Engine Active (Document Reader, Live APIs & Vector Memory Enabled).")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]: break
        reply = brain.process(user_input)
        print(f"{ASSISTANT_NAME}: {reply}")