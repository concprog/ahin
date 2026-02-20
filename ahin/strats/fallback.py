from typing import Dict, Any, Tuple
import random
from ahin.core import ResponseStrategyProtocol


class FallbackStrategy:
    """
    Fallback strategy that always matches and provides default responses.
    
    This strategy should be placed at the end of a router chain to handle
    any input that wasn't matched by more specific strategies.
    
    It returns generic "I didn't understand" type responses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default responses for unmatched input
        lang = self.config.get("assistant", {}).get("response_language", "hindi")
        
        self.responses = {
            "hindi": [
                "माफ़ कीजिये, मैं समझ नहीं पाया।",
                "क्या आप फिर से बोलेंगे?",
                "हम्म, यह मेरे समझ से बाहर है।",
                "थोड़ा और साफ़ बोलेंगे?"
            ],
            "english": [
                "Sorry, I didn't understand that.",
                "Could you please repeat that?",
                "I'm not sure what you mean.",
                "Could you say that more clearly?"
            ],
            "spanish": [
                "Lo siento, no entendí eso.",
                "¿Podrías repetir eso?",
                "No estoy seguro de lo que quieres decir.",
                "¿Podrías decirlo más claramente?"
            ],
            "french": [
                "Désolé, je n'ai pas compris.",
                "Pourriez-vous répéter s'il vous plaît?",
                "Je ne suis pas sûr de ce que vous voulez dire.",
                "Pourriez-vous le dire plus clairement?"
            ]
        }
        
        self.current_lang = lang

    def generate_response(self, text: str) -> Tuple[bool, str]:
        """
        Generate a fallback response.
        Always returns True (matched) since this is a catch-all strategy.
        
        Args:
            text: Input text (unused, since this is a fallback)
            
        Returns:
            Tuple of (True, fallback_response)
        """
        responses = self.responses.get(self.current_lang, self.responses["english"])
        response = random.choice(responses)
        return (True, response)
