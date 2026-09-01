import os

class Config:
    ASSISTANT_NAME = "BAT"
    MODEL_NAME = "bat-engine"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH = os.path.join(BASE_DIR, "brain_memory")