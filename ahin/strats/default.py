
from typing import Dict, Any
from ahin.core import ResponseStrategyProtocol

class DefaultResponseStrategy:
    """
    Default response strategy that echoes back the user's input.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate_response(self, text: str) -> str:
        """
        Generate a response based on the input text.
        For the default strategy, this simply echoes the text in the configured language.
        """
        lang = self.config["assistant"].get("response_language", "hindi")

        text = text.strip()
        text = "".join(list(text)) 
        responses = {
            "hindi": f"आपने कहा: {text}",
            "english": f"You said: {text}",
            "spanish": f"Dijiste: {text}",
            "french": f"Vous avez dit: {text}",
        }
        return responses.get(lang, f"You said: {text}")
