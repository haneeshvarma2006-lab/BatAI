import io
import contextlib
import traceback
import math
import datetime
import os

def execute_python_code(code: str) -> str:
    output = io.StringIO()
    # Pre-load common modules so the AI doesn't have to remember to import them
    safe_env = {
        "math": math,
        "datetime": datetime,
        "os": os,
        "__builtins__": __builtins__
    }
    
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exec(code, safe_env)
        
        result = output.getvalue()
        if not result.strip():
            return "Code executed successfully (no print output)."
        return result
    except Exception:
        return f"Code Execution Failed:\n{traceback.format_exc()}"