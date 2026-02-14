
from typing import Dict, Any, List, Tuple
import random
from ahin.core import ResponseStrategyProtocol

class ConversationalStrategy:
    """
    Response strategy that matches input text to Hindi commands/patterns 
    and selects a conversational response.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Dictionary of patterns to responses.
        # Keys are substrings to match in the "joined" text (no spaces).
        # Values are lists of possible responses.
        self.patterns: List[Tuple[str, List[str]]] = [
            ("नमस्ते", ["नमस्ते जी, कहिये क्या सेवा करूँ?", "नमस्ते, आज का दिन कैसा है?"]),
            ("कैसेहो", ["मैं ठीक हूँ, आप कैसे हैं?", "मैं तो एक मशीन हूँ, पर सब बढ़िया है।"]),
            ("क्याकररहेहो", ["मैं आपकी बात सुनने का इंतज़ार कर रहा हूँ।", "बस, आपके आदेश का पालन करने को तैयार हूँ।"]),
            ("तुमकौनहो", ["मैं आपका वॉइस असिस्टेंट हूँ।", "मैं एक AI हूँ जो आपकी मदद के लिए बनाया गया है।"]),
            ("नामक्याहै", ["मेरा नाम अहिन है।", "मुझे अहिन कहते हैं।"]),
            ("समयक्याहुआहै", ["माफ़ कीजिये, मुझे अभी समय देखने की अनुमति नहीं है।", "समय तो उड़ रहा है!"]), # Placeholder logic
            ("शुक्रिया", ["आपका स्वागत है!", "कोई बात नहीं, यह मेरा काम है।"]),
            ("धन्यवाद", ["आपका स्वागत है!", "खुशी हुई आपकी मदद करके।"]),
            ("टाटा", ["फिर मिलेंगे!", "अलविदा, अपना खयाल रखियेगा।"]),
            ("बाय", ["बाय बाय!", "फिर मिलते हैं।"]),
            # Add more patterns here matches against joined text
        ]
        
        self.default_responses = [
            "माफ़ कीजिये, मैं समझ नहीं पाया।",
            "क्या आप फिर से बोलेंगे?",
            "हम्म, यह मेरे समझ से बाहर है।",
            "थोड़ा और साफ़ बोलेंगे?"
        ]

    def generate_response(self, text: str) -> str:
        """
        Generate a response based on the input text.
        Matches patterns against joined text (spaces removed).
        """
        if not text:
            return ""
            
        # Join text to match patterns robustly against ASR variations
        cleaned_text = "".join(text.split())
        
        # Check for matches
        for pattern, responses in self.patterns:
            if pattern in cleaned_text:
                return random.choice(responses)
                
        # Fallback
        return random.choice(self.default_responses)
