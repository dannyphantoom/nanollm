import torch
from typing import Optional, Callable
from model import TransformerModel
from tokenizer import BPETokenizer
from inference.sampler import sample

class ChatHandler:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        on_token: Optional[Callable[[str], None]] = None
    ):
        # Load model and tokenizer
        self.model = TransformerModel.from_pretrained(model_path)
        self.tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
        
        # Move model to device
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Callback for streaming tokens
        self.on_token = on_token
        
        # System prompt to help guide the model's behavior
        self.system_prompt = """You are NanoLLM, a helpful and friendly AI assistant. 
You aim to provide accurate and helpful responses while being direct and concise.
You should always be truthful and admit when you're not sure about something."""
    
    def generate_response(self, message: str) -> str:
        # Combine system prompt with user message
        full_prompt = f"{self.system_prompt}\n\nUser: {message}\n\nAssistant:"
        
        # Generate response
        with torch.no_grad():
            response = sample(
                self.model,
                self.tokenizer,
                full_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                device=self.device,
                on_token=self.on_token
            )
        
        # Extract just the assistant's response
        response = response.split("Assistant:")[-1].strip()
        return response
    
    @classmethod
    def from_config(cls, config_path: str) -> "ChatHandler":
        """Create a ChatHandler from a config file"""
        import json
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(
            model_path=config["model_path"],
            tokenizer_path=config["tokenizer_path"],
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            max_new_tokens=config.get("max_new_tokens", 100),
            temperature=config.get("temperature", 0.8),
            top_p=config.get("top_p", 0.9)
        ) 