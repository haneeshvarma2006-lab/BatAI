from core.brain import CognitiveBrain
from core.config import Config

def main():
    print(f"=== {Config.ASSISTANT_NAME} V2 Agent Initialized ===")
    print("Type your command (or 'exit' to quit).\n")
    
    brain = CognitiveBrain()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print(f"Shutting down {Config.ASSISTANT_NAME}. Goodbye!")
                break
                
            if not user_input:
                continue
                
            print(f"\n[{Config.ASSISTANT_NAME} is thinking...]")
            response = brain.think_and_act(user_input)
            print(f"\n{Config.ASSISTANT_NAME}: {response}\n")
            
        except KeyboardInterrupt:
            print(f"\nShutting down {Config.ASSISTANT_NAME}...")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")

if __name__ == "__main__":
    main()