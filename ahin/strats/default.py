
from typing import Dict, Any, Tuple
from ahin.core import ResponseStrategyProtocol

class ConversationalStrategy:
    """
    Default response strategy that echoes back the user's input.
    This strategy always matches (returns True).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate_response(self, text: str) -> Tuple[bool, str]:
        """
        Generate a response based on the input text.
        For the default strategy, this simply echoes the text in the configured language.
        Always returns True (matched) since this is a catch-all echo strategy.
        
        Returns:
            Tuple of (True, echo_response)
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
        response = responses.get(lang, f"You said: {text}")
        return (True, response)
