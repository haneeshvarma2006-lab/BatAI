import json
import ollama
from core.config import Config
from memory.chroma_cloud import MemoryCloud
from tools.system_tools import get_system_stats, open_application, find_file_location, search_and_open_file, automate_typing
from tools.code_exec import execute_python_code

class CognitiveBrain:
    def __init__(self):
        self.model = Config.MODEL_NAME
        self.memory = MemoryCloud()
        self.working_memory = []
        
        self.tool_map = {
            "get_system_stats": get_system_stats,
            "open_application": open_application,
            "find_file_location": find_file_location,
            "search_and_open_file": search_and_open_file,
            "automate_typing": automate_typing,
            "execute_python_code": execute_python_code,
            "save_user_fact": self.memory.save_user_fact
        }
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_system_stats",
                    "description": "Returns live hardware metrics including CPU, RAM, and Battery percentage."
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_file_location",
                    "description": "Searches disk drives for the exact path of a file or script on the system.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "The file name to locate, e.g., 'main.py'"}
                        },
                        "required": ["filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_and_open_file",
                    "description": "Locates and opens a file in its default desktop application.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Name of the file to open"}
                        },
                        "required": ["filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_python_code",
                    "description": "Mandatory for all mathematical calculations, factorials, algorithms, or data processing. Runs valid Python code in a REPL and returns stdout.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Executable Python code with print() statements"}
                        },
                        "required": ["code"]
                    }
                }
            }
        ]

    def think_and_act(self, user_input: str) -> str:
        recalled_context = self.memory.search_all(user_input)
        
        system_instruction = (
            f"You are {Config.ASSISTANT_NAME}, an autonomous personal AI system running on Windows.\n"
            "STRICT OPERATIONAL RULES:\n"
            "1. NEVER calculate factorials, powers, or complex math mentally. You MUST call 'execute_python_code' with valid Python code.\n"
            "2. NEVER invent file paths or assume Linux directories. Call 'find_file_location' to search the actual filesystem.\n"
            "3. For hardware status, call 'get_system_stats'.\n\n"
            f"CONTEXT FROM VECTOR MEMORY:\n{recalled_context}"
        )

        messages = [{"role": "system", "content": system_instruction}]
        messages.extend(self.working_memory[-6:])
        messages.append({"role": "user", "content": user_input})

        for step in range(5):
            response = ollama.chat(model=self.model, messages=messages, tools=self.tools)
            message = response["message"]
            messages.append(message)

            if not message.get("tool_calls"):
                final_answer = message["content"]
                self.working_memory.append({"role": "user", "content": user_input})
                self.working_memory.append({"role": "assistant", "content": final_answer})
                return final_answer

            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]

                print(f"\n[Tool Execution] Invoking `{func_name}` with parameters: {arguments}")

                if func_name in self.tool_map:
                    try:
                        tool_result = self.tool_map[func_name](**arguments)
                    except Exception as e:
                        tool_result = f"Tool Execution Error: {e}"
                else:
                    tool_result = f"Tool '{func_name}' not recognized."

                print(f"[Tool Observation] Result: {tool_result}\n")

                messages.append({
                    "role": "tool",
                    "content": str(tool_result)
                })

        return "Task could not be completed within the step limit."