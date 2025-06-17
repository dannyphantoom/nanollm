import os
import sys
import json
from pathlib import Path
from chat_window import ModernChatWindow
from chat_handler import ChatHandler

def main():
    # Get the project root directory
    root_dir = Path(__file__).parent.parent
    
    # Load config
    config_path = os.path.join(root_dir, "GUI", "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update paths to be relative to root
    config["model_path"] = os.path.join(root_dir, config["model_path"])
    config["tokenizer_path"] = os.path.join(root_dir, config["tokenizer_path"])
    
    try:
        # Initialize chat handler
        chat_handler = ChatHandler.from_config(config_path)
        
        # Define callback for handling messages
        def handle_message(message: str):
            try:
                response = chat_handler.generate_response(message)
                window.add_message("Assistant", response)
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                window.add_message("System", error_msg)
        
        # Create and start chat window
        window = ModernChatWindow(handle_message)
        
        # Add welcome message
        welcome_msg = """Welcome to NanoLLM Chat! 
I'm your AI assistant, trained to help you with various tasks.
Feel free to ask me anything!"""
        window.add_message("Assistant", welcome_msg)
        
        # Start the GUI
        window.start()
        
    except Exception as e:
        print(f"Error initializing chat: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 