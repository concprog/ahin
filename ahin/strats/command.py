
from typing import Dict, Any, List, Tuple
import random
from ahin.core import ResponseStrategyProtocol

import urllib.request
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Callable, Union

class ConversationalStrategy:
    """
    Response strategy that matches input text to Hindi commands/patterns 
    and selects a conversational response, enriched with 15 public API apps.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Dictionary of patterns to responses/functions.
        self.patterns: List[Tuple[str, Union[List[str], Callable[[], str]]]] = [
            ("नमस्ते", ["नमस्ते जी, कहिये क्या सेवा करूँ?", "नमस्ते, आज का दिन कैसा है?"]),
            ("कैसेहो", ["मैं ठीक हूँ, आप कैसे हैं?", "मैं तो एक मशीन हूँ, पर सब बढ़िया है।"]),
            ("क्याकररहेहो", ["मैं आपकी बात सुनने का इंतज़ार कर रहा हूँ।", "बस, आपके आदेश का पालन करने को तैयार हूँ।"]),
            ("तुमकौनहो", ["मैं आपका वॉइस असिस्टेंट हूँ।", "मैं एक AI हूँ जो आपकी मदद के लिए बनाया गया है।"]),
            ("नामक्याहै", ["मेरा नाम अहिन है।", "मुझे अहिन कहते हैं।"]),
            ("शुक्रिया", ["आपका स्वागत है!", "कोई बात नहीं, यह मेरा काम है।"]),
            ("धन्यवाद", ["आपका स्वागत है!", "खुशी हुई आपकी मदद करके।"]),
            ("टाटा", ["फिर मिलेंगे!", "अलविदा, अपना खयाल रखियेगा।"]),
            ("बाय", ["बाय बाय!", "फिर मिलते हैं।"]),
            
            # 15 Public API commands
            ("मौसम", self.get_weather),
            ("चुटकुला", self.get_joke),
            ("मजाक", self.get_joke),
            ("फैक्ट", self.get_fact),
            ("तथ्य", self.get_fact),
            ("समय", self.get_time),
            ("तारीख", self.get_date),
            ("सलाह", self.get_advice),
            ("बिटकॉइन", self.get_bitcoin),
            ("बिल्ली", self.get_cat_fact),
            ("अंतरिक्ष", self.get_iss_location),
            ("कुत्ते", self.get_dog_status),
            ("आईपी", self.get_ip),
            ("उम्र", self.get_agify),
            ("लिंग", self.get_genderize),
            ("राष्ट्रीयता", self.get_nationalize),
            ("गणित", self.get_number_fact)
        ]
        
        self.default_responses = [
            "माफ़ कीजिये, मैं समझ नहीं पाया।",
            "क्या आप फिर से बोलेंगे?",
            "हम्म, यह मेरे समझ से बाहर है।",
            "थोड़ा और साफ़ बोलेंगे?"
        ]

    def _fetch_json(self, url: str) -> dict:
        """Helper to fetch public APIs securely without dependencies."""
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            print(f"API Error fetching {url}: {e}")
            return {}

    def get_weather(self) -> str:
        data = self._fetch_json("https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&current_weather=true")
        if data and "current_weather" in data:
            temp = data["current_weather"]["temperature"]
            speed = data["current_weather"]["windspeed"]
            return f"अभी दिल्ली का तापमान {temp} डिग्री सेल्सियस है और हवा की गति {speed} किलोमीटर प्रति घंटा है।"
        return "माफ़ कीजिये, मौसम की जानकारी नहीं मिल रही है।"

    def get_joke(self) -> str:
        data = self._fetch_json("https://official-joke-api.appspot.com/random_joke")
        if data:
            return f"एक चुटकुला सुनिए: {data.get('setup')} ... {data.get('punchline')}."
        return "मुझे अभी कोई चुटकुला याद नहीं आ रहा।"

    def get_fact(self) -> str:
        data = self._fetch_json("https://uselessfacts.jsph.pl/api/v2/facts/random")
        if data and "text" in data:
            return f"क्या आप जानते हैं? {data['text']}"
        return "मेरे पास अभी कोई नया तथ्य नहीं है।"

    def get_time(self) -> str:
        now = datetime.now()
        return f"अभी समय हुआ है {now.strftime('%I बज कर %M मिनट')}।"

    def get_date(self) -> str:
        now = datetime.now()
        return f"आज की तारीख है {now.strftime('%d %B, %Y')}।"

    def get_advice(self) -> str:
        data = self._fetch_json("https://api.adviceslip.com/advice")
        if data and "slip" in data:
            return f"मेरी सलाह है: {data['slip']['advice']}"
        return "मुझे समझ नहीं आ रहा कि क्या सलाह दूँ।"

    def get_bitcoin(self) -> str:
        data = self._fetch_json("https://api.coindesk.com/v1/bpi/currentprice.json")
        if data and "bpi" in data:
            price = data["bpi"]["USD"]["rate"]
            return f"अभी एक बिटकॉइन की कीमत लगभग {price} अमेरिकी डॉलर है।"
        return "बिटकॉइन की कीमत अभी उपलब्ध नहीं है।"

    def get_cat_fact(self) -> str:
        data = self._fetch_json("https://catfact.ninja/fact")
        if data and "fact" in data:
            return f"बिल्लियों के बारे में एक तथ्य: {data['fact']}"
        return "बिल्लियों के बारे में अभी कोई जानकारी नहीं है।"

    def get_iss_location(self) -> str:
        data = self._fetch_json("http://api.open-notify.org/iss-now.json")
        if data and "iss_position" in data:
            pos = data["iss_position"]
            return f"अंतर्राष्ट्रीय अंतरिक्ष स्टेशन अभी अक्षांश {pos['latitude']} और देशांतर {pos['longitude']} पर है।"
        return "अंतरिक्ष स्टेशन की लोकेशन नहीं मिल पा रही।"

    def get_dog_status(self) -> str:
        data = self._fetch_json("https://dog.ceo/api/breeds/image/random")
        if data and data.get("status") == "success":
            return "मैंने कुत्तों के डेटाबेस में एक नयी तस्वीर ढूँढी है, लेकिन मैं आपको दिखा नहीं सकता!"
        return "कुत्तों का सर्वर अभी व्यस्त है।"

    def get_ip(self) -> str:
        data = self._fetch_json("https://api.ipify.org?format=json")
        if data and "ip" in data:
            return f"आपका सार्वजनिक आईपी एड्रेस {data['ip']} है।"
        return "मैं आपका आईपी नहीं ढूँढ पा रहा।"

    def get_agify(self) -> str:
        data = self._fetch_json("https://api.agify.io?name=ahin")
        if data and "age" in data and data["age"]:
            return f"अहिन नाम के लोगों की औसत उम्र {data['age']} साल होती है।"
        return "नाम से पता चला कि उम्र का अनुमान नहीं लग पाया।"

    def get_genderize(self) -> str:
        data = self._fetch_json("https://api.genderize.io?name=ahin")
        if data and "gender" in data and data["gender"]:
            return f"अहिन नाम ज़्यादातर {data['gender']} का होता है।"
        return "नाम से लिंग का अनुमान नहीं लग पाया।"

    def get_nationalize(self) -> str:
        data = self._fetch_json("https://api.nationalize.io?name=ahin")
        if data and "country" in data and len(data["country"]) > 0:
            country_id = data["country"][0]["country_id"]
            return f"अहिन नाम के लोगों के {country_id} देश से होने की सबसे अधिक संभावना है।"
        return "किस देश का नाम है, यह पता नहीं चला।"

    def get_number_fact(self) -> str:
        data = self._fetch_json("http://numbersapi.com/random/trivia?json")
        if data and "text" in data:
            return f"गणित का एक तथ्य: {data['text']}"
        return "अभी कोई नंबर का तथ्य याद नहीं आ रहा।"

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
        for pattern, response_data in self.patterns:
            if pattern in cleaned_text:
                if callable(response_data):
                    return response_data()
                return random.choice(response_data)
                
        # Fallback
        return random.choice(self.default_responses)
