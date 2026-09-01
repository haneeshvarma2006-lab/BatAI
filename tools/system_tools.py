import os
import psutil
import subprocess
import pyautogui
import time

def get_system_stats() -> str:
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    battery = psutil.sensors_battery()
    bat_str = f"{battery.percent}%" if battery else "Desktop/AC"
    return f"CPU Usage: {cpu}% | RAM Usage: {ram}% | Battery Level: {bat_str}"

def open_application(app_name: str) -> str:
    apps = {"notepad": "notepad", "chrome": "start chrome", "explorer": "explorer", "calculator": "calc"}
    target = app_name.lower().strip()
    if target in apps:
        subprocess.Popen(apps[target], shell=True)
        return f"Successfully opened {target}."
    try:
        subprocess.Popen(f"start {target}", shell=True)
        return f"Attempted to launch {target}."
    except Exception as e:
        return f"Failed to launch application: {e}"

def find_file_location(filename: str) -> str:
    """Searches the D: drive and returns the exact file paths found."""
    found_paths = set() # Using a set prevents duplicate paths!
    search_roots = ["D:\\BatAI", "D:\\"]
    
    for base in search_roots:
        if not os.path.exists(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if filename.lower() == f.lower() or filename.lower() in f.lower():
                    found_paths.add(os.path.join(root, f))
                    if len(found_paths) >= 3:
                        break
            if len(found_paths) >= 3:
                break
                
    if found_paths:
        return "Found matching file(s):\n" + "\n".join(list(found_paths))
    return f"No file named '{filename}' found on D: drive."

def search_and_open_file(filename: str) -> str:
    """Searches the D: drive for a file and launches it."""
    for base in ["D:\\BatAI", "D:\\"]:
        if not os.path.exists(base):
            continue
        for root, _, files in os.walk(base):
            if filename.lower() in [f.lower() for f in files]:
                file_path = os.path.join(root, filename)
                os.startfile(file_path)
                return f"Successfully opened: {file_path}"
    return f"Could not find {filename} to open on D: drive."

def automate_typing(text: str) -> str:
    time.sleep(3)
    pyautogui.write(text, interval=0.05)
    return "Completed simulated typing."