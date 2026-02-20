from typing import Dict, Any, Tuple
import os
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None # type: ignore

from ahin.core import ResponseStrategyProtocol

import urllib.request
import json
from datetime import datetime

class ConversationalStrategy:
    """
    Response strategy that uses a local LLM (OpenAI compatible) to generate responses.
    This version uses OpenAI Tool Calling to execute 15 public APIs.
    This strategy always matches (returns True) and attempts to generate a response.
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
            "You use tools to fetch real-world data to answer user questions. "
            "If a tool gives you English data, translate and summarize it in conversational Hindi, keep it short - six to seven words and max two sentences. NO MARKDOWN or formatting, simple hindi sentences"
            " Correct the user's input if it is misspelled or grammatically incorrect Hindi, if that is not poossible, infer the meaning and respond accordingly."
        )+"""
        एक धुआंधार, मजाकिया और मददगार हिंदी वॉइस असिस्टेंट।

    तुम्हारी शैली: बातचीत-शैली (जैसे दोस्त के साथ), संक्षिप्त, स्वाभाविक और गलतियों को नज़रअंदाज़ करते हुए।

    तुम्हारा काम:
        अस्पष्ट/गलत शब्दों को सही-सही समझना (जैसे "मैने कहा था" → "मैंने कहा था").
        चोटे-चोटे सवालों (जैसे "क्या मौसम है?", "कौन सा गाना सुनाऊँ?") के लिए तुरंत और सरल जवाब देना।
        हिंदी (देवनागरी) में सहज और रोचक तरीके से जवाब देना।

    तुम्हारी पर्सनालिटी:
        चेन्नई में रहते हो, श्रेया घोषाल, अरिजीत सिंह और किशोर कुमार के फैन हो।
        मौसम: आज (17 फरवरी, 2026) धूप + आंशिक बादल, 28°C (मैक्स) / 22°C (मिन).
        मजाकिया और संवेदनशील हो, लेकिन सही समय पर गंभीर भी।

    अतिरिक्त निर्देश:
        गलतियाँ न सुधारो, बस सही माने को समझो और बातचीत को आगे बढ़ाओ।
        उदाहरण:
            "अब तक का सबसे अच्छा गाना?" → "अब तो ‘तू ही मेरा दिवाना’ (अरिजीत) या ‘जय हो’ (श्रेया)!"
            "आज का मौसम?" → "चेन्नई में धूप + बादल, 28°C तक! सनस्क्रीन लेना न भूलना 😎"
            "क्या खाना बनाऊँ?" → "अब तो चेन्नई का दोसा या दही भल्ले!"
    """
        self.system_prompt = llm_config.get("system_prompt", default_system_prompt)
        
        self.tools = [
            {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather and wind speed for Delhi."}},
            {"type": "function", "function": {"name": "get_joke", "description": "Get a random joke."}},
            {"type": "function", "function": {"name": "get_fact", "description": "Get a random useless fact."}},
            {"type": "function", "function": {"name": "get_time", "description": "Get the current local time."}},
            {"type": "function", "function": {"name": "get_date", "description": "Get the current local date."}},
            {"type": "function", "function": {"name": "get_advice", "description": "Get random life advice."}},
            {"type": "function", "function": {"name": "get_bitcoin", "description": "Get the current Bitcoin price in USD."}},
            {"type": "function", "function": {"name": "get_cat_fact", "description": "Get a random fact about cats."}},
            {"type": "function", "function": {"name": "get_iss_location", "description": "Get the current latitude and longitude of the International Space Station (ISS)."}},
            {"type": "function", "function": {"name": "get_dog_status", "description": "Get a status message about random dog pictures."}},
            {"type": "function", "function": {"name": "get_ip", "description": "Get the user's public IP address."}},
            {"type": "function", "function": {"name": "get_agify", "description": "Get the estimated age for the name 'Ahin'."}},
            {"type": "function", "function": {"name": "get_genderize", "description": "Get the estimated gender for the name 'Ahin'."}},
            {"type": "function", "function": {"name": "get_nationalize", "description": "Get the estimated nationality for the name 'Ahin'."}},
            {"type": "function", "function": {"name": "get_number_fact", "description": "Get a random trivia fact about a number."}}
        ]
        
        self.available_functions = {
            "get_weather": self.get_weather,
            "get_joke": self.get_joke,
            "get_fact": self.get_fact,
            "get_time": self.get_time,
            "get_date": self.get_date,
            "get_advice": self.get_advice,
            "get_bitcoin": self.get_bitcoin,
            "get_cat_fact": self.get_cat_fact,
            "get_iss_location": self.get_iss_location,
            "get_dog_status": self.get_dog_status,
            "get_ip": self.get_ip,
            "get_agify": self.get_agify,
            "get_genderize": self.get_genderize,
            "get_nationalize": self.get_nationalize,
            "get_number_fact": self.get_number_fact
        }

    def _fetch_json(self, url: str) -> str:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.read().decode()
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_weather(self) -> str:
        return self._fetch_json("https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&current_weather=true")

    def get_joke(self) -> str:
        return self._fetch_json("https://official-joke-api.appspot.com/random_joke")

    def get_fact(self) -> str:
        return self._fetch_json("https://uselessfacts.jsph.pl/api/v2/facts/random")

    def get_time(self) -> str:
        return json.dumps({"time": datetime.now().strftime('%I:%M %p')})

    def get_date(self) -> str:
        return json.dumps({"date": datetime.now().strftime('%d %B, %Y')})

    def get_advice(self) -> str:
        return self._fetch_json("https://api.adviceslip.com/advice")

    def get_bitcoin(self) -> str:
        return self._fetch_json("https://api.coindesk.com/v1/bpi/currentprice.json")

    def get_cat_fact(self) -> str:
        return self._fetch_json("https://catfact.ninja/fact")

    def get_iss_location(self) -> str:
        return self._fetch_json("http://api.open-notify.org/iss-now.json")

    def get_dog_status(self) -> str:
        return self._fetch_json("https://dog.ceo/api/breeds/image/random")

    def get_ip(self) -> str:
        return self._fetch_json("https://api.ipify.org?format=json")

    def get_agify(self) -> str:
        return self._fetch_json("https://api.agify.io?name=ahin")

    def get_genderize(self) -> str:
        return self._fetch_json("https://api.genderize.io?name=ahin")

    def get_nationalize(self) -> str:
        return self._fetch_json("https://api.nationalize.io?name=ahin")

    def get_number_fact(self) -> str:
        return self._fetch_json("http://numbersapi.com/random/trivia?json")

    def generate_response(self, text: str) -> Tuple[bool, str]:
        """
        Generate a response using the LLM with tool calling support.
        """
        if not text or not text.strip():
            return (False, "")
            
        try:
            messages = [
                {"role": "user", "content": self.system_prompt+text},
                # {"role": "user", "content": text}
            ]
            
            # Step 1: Send initial request with tools
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.5,
                max_tokens=1024,
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            # Step 2: Check if model wanted to call a tool
            if tool_calls:
                messages.append(response_message)  # Extend conversation with assistant's reply
                
                # Step 3: Call the tools
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = self.available_functions.get(function_name)
                    
                    if function_to_call:
                        print(f"[{function_name}] tool called by LLM...")
                        function_response = function_to_call()
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }
                        )
                
                # Step 4: Loop back to LLM with the tool responses to generate final speech
                second_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024,
                )
                
                full_response = second_response.choices[0].message.content
                
                return full_response
            else:
                # No tool called, just return the standard text response
                content = response_message.content or "माफ़ कीजिये, मैं जवाब नहीं दे पा रहा हूँ।"
                print(content)
                return content.strip()
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return "माफ़ कीजिये, अभी मैं जवाब नहीं दे पा रहा हूँ।"
