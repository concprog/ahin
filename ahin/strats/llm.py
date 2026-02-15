
from typing import Dict, Any
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None # type: ignore

from ahin.core import ResponseStrategyProtocol

class LLMStrategy:
    """
    Response strategy that uses a local LLM (OpenAI compatible) to generate responses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        if OpenAI is None:
            raise ImportError("Please install 'openai' package: uv add openai")
            
        llm_config = config.get("llm", {})
        self.client = OpenAI(
            base_url=llm_config.get("base_url", "http://localhost:8000/v1"),
            api_key=llm_config.get("api_key", "EMPTY"),
        )
        self.model = llm_config.get("model", "gpt-3.5-turbo")
        
        # Enhanced system prompt for Hindi ASR correction and response
        default_system_prompt = (
            "You are Ahin, a smart and helpful voice assistant who speaks in Hindi. "
            "The user input is transcribed from speech and may contain errors, spelling mistakes, "
            "or be phonetically approximated (garbled). "
            "Your goal is to: "
            "1. Smartly infer the user's actual intent effectively fixing the garbled text. "
            "2. Provide a helpful, concise, and conversational response in Hindi (Devanagari script). "
            "Do not mention the errors, just answer the user naturally."
        )
        self.system_prompt = llm_config.get("system_prompt", default_system_prompt)

    def generate_response(self, text: str) -> str:
        """
        Generate a response using the LLM.
        """
        if not text or not text.strip():
            return ""
            
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=150, # Keep responses concise for voice
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return "माफ़ कीजिये, अभी मैं जवाब नहीं दे पा रहा हूँ।" # "Sorry, I cannot answer right now."
