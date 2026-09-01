import chromadb
import datetime
from core.config import Config

class MemoryCloud:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.DB_PATH)
        self.knowledge = self.client.get_or_create_collection(name="knowledge")
        self.user_profile = self.client.get_or_create_collection(name="user_profile")
        self.documents = self.client.get_or_create_collection(name="local_documents")

    def save_user_fact(self, fact: str) -> str:
        doc_id = f"fact_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.user_profile.add(
            documents=[fact],
            metadatas=[{"type": "user_preference"}],
            ids=[doc_id]
        )
        return f"[{Config.ASSISTANT_NAME}] Fact saved to memory."
        
    def search_all(self, query: str) -> str:
        u_res = self.user_profile.query(query_texts=[query], n_results=1)
        k_res = self.knowledge.query(query_texts=[query], n_results=1)
        d_res = self.documents.query(query_texts=[query], n_results=1)
        
        context = ""
        if u_res.get("documents") and u_res["documents"][0]:
            context += "USER PROFILE MEMORY:\n" + "\n".join(u_res["documents"][0]) + "\n\n"
        if k_res.get("documents") and k_res["documents"][0]:
            context += "STORED KNOWLEDGE:\n" + "\n".join(k_res["documents"][0]) + "\n\n"
        if d_res.get("documents") and d_res["documents"][0]:
            context += "DOCUMENT EXCERPTS:\n" + "\n".join(d_res["documents"][0]) + "\n\n"
            
        return context.strip()