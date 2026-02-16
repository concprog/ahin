from typing import Dict, Any
import os
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None # type: ignore

from ahin.core import ResponseStrategyProtocol

class ConversationalStrategy:
    """
    Response strategy that uses a local LLM (OpenAI compatible) to generate responses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        load_dotenv()
        self.config = config
        if OpenAI is None:
            raise ImportError("Please install 'openai' package: uv add openai")
            
        llm_config = config.get("llm", {})
        
        # NVIDIA API Configuration
        base_url = llm_config.get("base_url", "https://integrate.api.nvidia.com/v1")
        api_key = llm_config.get("api_key") or os.getenv("NVIDIA_API_KEY")
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = llm_config.get("model", "nvidia/nemotron-4-mini-hindi-4b-instruct")
        
        # Enhanced system prompt for Hindi ASR correction and response
        default_system_prompt = (
            "You are Ahin, a smart and helpful voice assistant who speaks in Hindi. "
            "The user input is transcribed from speech and may contain errors, spelling mistakes, "
            "यूज़र का इनपुट भाषण से लिया गया है और इसमें त्रुटियां, वर्तनी की गलतियां, "
            "या ध्वन्यात्मक रूप से अनुमानित (अस्पष्ट) शब्द हो सकते हैं। "
            "तुम्हारा लक्ष्य है: \n"
            "1. समझदारी से यूज़र के असली इरादे का अनुमान लगाना और अस्पष्ट टेक्स्ट को ठीक करना। \n"
            "2. हिंदी (देवनागरी लिपि) में एक मददगार, संक्षिप्त और बातचीत की शैली में जवाब देना। \n"
            "गलतियों का जिक्र मत करना, बस स्वाभाविक रूप से जवाब देना।"
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
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                top_p=1,
                max_tokens=1024,
                stream=True
            )
            
            full_response = []
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response.append(content)
            print() # Ensure newline after stream
            
            return "".join(full_response).strip()
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return "माफ़ कीजिये, अभी मैं जवाब नहीं दे पा रहा हूँ।" # "Sorry, I cannot answer right now."
