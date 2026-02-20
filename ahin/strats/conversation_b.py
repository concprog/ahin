from typing import Dict, Any, List, Tuple
import random
import datetime
import platform
import subprocess
import os
from ahin.core import ResponseStrategyProtocol

class ConversationalStrategy:
    """
    Response strategy that matches input text to Hindi commands/patterns 
    and selects a conversational response.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        self.patterns: List[Tuple[str, List[str]]] = [
            # ── Greetings / Identity ───────────────────────────────────────────
            ("नमस्ते",          ["नमस्ते जी, कहिये क्या सेवा करूँ?", "नमस्ते, आज का दिन कैसा है?"]),
            ("कैसेहो",          ["मैं ठीक हूँ, आप कैसे हैं?", "मैं तो एक मशीन हूँ, पर सब बढ़िया है।"]),
            ("क्याकररहेहो",    ["मैं आपकी बात सुनने का इंतज़ार कर रहा हूँ।", "बस, आपके आदेश का पालन करने को तैयार हूँ।"]),
            ("तुमकौनहो",       ["मैं आपका वॉइस असिस्टेंट हूँ।", "मैं एक AI हूँ जो आपकी मदद के लिए बनाया गया है।"]),
            ("नामक्याहै",      ["मेरा नाम अहिन है।", "मुझे अहिन कहते हैं।"]),

            # ── Farewells / Thanks ─────────────────────────────────────────────
            ("शुक्रिया",       ["आपका स्वागत है!", "कोई बात नहीं, यह मेरा काम है।"]),
            ("धन्यवाद",        ["आपका स्वागत है!", "खुशी हुई आपकी मदद करके।"]),
            ("टाटा",           ["फिर मिलेंगे!", "अलविदा, अपना खयाल रखियेगा।"]),
            ("बाय",            ["बाय बाय!", "फिर मिलते हैं।"]),

            # ══════════════════════════════════════════════════════════════════
            # 10 OFFLINE COMMANDS
            # ══════════════════════════════════════════════════════════════════

            # 1. Current time  ─────────────────────────────────────────────────
            #    Trigger: "समय क्या हुआ है" / "अभी क्या बजे हैं"
            ("समयक्याहुआहै",   [self._get_time]),
            ("क्याबजेहैं",     [self._get_time]),

            # 2. Current date  ─────────────────────────────────────────────────
            #    Trigger: "आज की तारीख क्या है"
            ("तारीखक्याहै",    [self._get_date]),
            ("आजकीतारीख",      [self._get_date]),

            # 3. Day of week  ──────────────────────────────────────────────────
            #    Trigger: "आज कौन सा दिन है"
            ("कौनसादिनहै",     [self._get_day]),

            # 4. Set a timer  ──────────────────────────────────────────────────
            #    Trigger: "टाइमर लगाओ" (no true countdown without a loop,
            #    but we acknowledge and log it for the caller to handle)
            ("टाइमरलगाओ",      [self._set_timer]),
            ("अलार्मलगाओ",     [self._set_timer]),

            # 5. Battery status  ───────────────────────────────────────────────
            #    Trigger: "बैटरी कितनी है"
            ("बैटरीकितनीहै",   [self._get_battery]),
            ("बैटरीस्टेटस",    [self._get_battery]),

            # 6. Volume up  ────────────────────────────────────────────────────
            #    Trigger: "आवाज़ बढ़ाओ"
            ("आवाज़बढ़ाओ",      [self._volume_up]),
            ("वॉल्यूमबढ़ाओ",   [self._volume_up]),

            # 7. Volume down  ──────────────────────────────────────────────────
            #    Trigger: "आवाज़ कम करो"
            ("आवाज़कमकरो",     [self._volume_down]),
            ("वॉल्यूमकमकरो",  [self._volume_down]),

            # 8. Mute  ─────────────────────────────────────────────────────────
            #    Trigger: "म्यूट करो" / "चुप करो"
            ("म्यूटकरो",       [self._mute]),
            ("चुपकरो",         [self._mute]),

            # 9. Take a screenshot  ────────────────────────────────────────────
            #    Trigger: "स्क्रीनशॉट लो"
            ("स्क्रीनशॉटलो",   [self._take_screenshot]),
            ("स्क्रीनशॉट",     [self._take_screenshot]),

            # 10. Lock / sleep the screen  ─────────────────────────────────────
            #     Trigger: "स्क्रीन बंद करो" / "कंप्यूटर लॉक करो"
            ("स्क्रीनबंदकरो",  [self._lock_screen]),
            ("लॉककरो",         [self._lock_screen]),
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Command handlers  (all offline)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_time(self) -> str:
        now = datetime.datetime.now()
        return f"अभी {now.hour} बजकर {now.minute} मिनट हुए हैं।"

    def _get_date(self) -> str:
        today = datetime.date.today()
        months = [
            "", "जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून",
            "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"
        ]
        return f"आज {today.day} {months[today.month]} {today.year} है।"

    def _get_day(self) -> str:
        days = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]
        return f"आज {days[datetime.date.today().weekday()]} है।"

    def _set_timer(self) -> str:
        # Actual countdown must be driven by the caller;
        # we return an acknowledgement and signal via a known prefix.
        return "TIMER:60:ठीक है, एक मिनट का टाइमर लगा दिया।"

    def _get_battery(self) -> str:
        try:
            import psutil  # optional lightweight dependency
            battery = psutil.sensors_battery()
            if battery is None:
                return "इस डिवाइस में बैटरी नहीं मिली।"
            status = "चार्ज हो रही है" if battery.power_plugged else "चार्ज नहीं हो रही"
            return f"बैटरी {int(battery.percent)} प्रतिशत है, {status}।"
        except ImportError:
            return "बैटरी जानकारी के लिए psutil इंस्टॉल करें।"

    def _volume_up(self) -> str:
        system = platform.system()
        try:
            if system == "Linux":
                subprocess.run(["amixer", "-q", "sset", "Master", "10%+"], check=True)
            elif system == "Darwin":
                subprocess.run(["osascript", "-e",
                                 "set volume output volume (output volume of (get volume settings) + 10)"],
                                check=True)
            elif system == "Windows":
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                current = volume.GetMasterVolumeLevelScalar()
                volume.SetMasterVolumeLevelScalar(min(1.0, current + 0.1), None)
            return "आवाज़ बढ़ा दी गई।"
        except Exception as e:
            return f"आवाज़ बढ़ाने में समस्या हुई: {e}"

    def _volume_down(self) -> str:
        system = platform.system()
        try:
            if system == "Linux":
                subprocess.run(["amixer", "-q", "sset", "Master", "10%-"], check=True)
            elif system == "Darwin":
                subprocess.run(["osascript", "-e",
                                 "set volume output volume (output volume of (get volume settings) - 10)"],
                                check=True)
            elif system == "Windows":
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                current = volume.GetMasterVolumeLevelScalar()
                volume.SetMasterVolumeLevelScalar(max(0.0, current - 0.1), None)
            return "आवाज़ कम कर दी गई।"
        except Exception as e:
            return f"आवाज़ कम करने में समस्या हुई: {e}"

    def _mute(self) -> str:
        system = platform.system()
        try:
            if system == "Linux":
                subprocess.run(["amixer", "-q", "sset", "Master", "toggle"], check=True)
            elif system == "Darwin":
                subprocess.run(["osascript", "-e", "set volume with output muted"], check=True)
            elif system == "Windows":
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMute(1, None)
            return "आवाज़ म्यूट कर दी गई।"
        except Exception as e:
            return f"म्यूट करने में समस्या हुई: {e}"

    def _take_screenshot(self) -> str:
        system = platform.system()
        path = os.path.expanduser(f"~/screenshot_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
        try:
            if system == "Linux":
                subprocess.run(["scrot", path], check=True)
            elif system == "Darwin":
                subprocess.run(["screencapture", "-x", path], check=True)
            elif system == "Windows":
                subprocess.run(
                    ["powershell", "-command",
                     f"Add-Type -AssemblyName System.Windows.Forms; "
                     f"[System.Windows.Forms.Screen]::PrimaryScreen | ForEach-Object {{ "
                     f"$bmp = New-Object System.Drawing.Bitmap($_.Bounds.Width,$_.Bounds.Height); "
                     f"$g = [System.Drawing.Graphics]::FromImage($bmp); "
                     f"$g.CopyFromScreen($_.Bounds.Location,[System.Drawing.Point]::Empty,$_.Bounds.Size); "
                     f"$bmp.Save('{path}') }}"],
                    check=True
                )
            return f"स्क्रीनशॉट ले लिया गया और {path} पर सेव हो गया।"
        except Exception as e:
            return f"स्क्रीनशॉट लेने में समस्या हुई: {e}"

    def _lock_screen(self) -> str:
        system = platform.system()
        try:
            if system == "Linux":
                subprocess.run(["loginctl", "lock-session"], check=True)
            elif system == "Darwin":
                subprocess.run([
                    "osascript", "-e",
                    'tell application "System Events" to keystroke "q" '
                    'using {command down, control down}'
                ], check=True)
            elif system == "Windows":
                subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=True)
            return "स्क्रीन लॉक कर दी गई।"
        except Exception as e:
            return f"स्क्रीन लॉक करने में समस्या हुई: {e}"

    # ──────────────────────────────────────────────────────────────────────────
    # Core matching logic
    # ──────────────────────────────────────────────────────────────────────────

    def generate_response(self, text: str) -> Tuple[bool, str]:
        """
        Generate a response based on the input text.
        Matches patterns against joined text (spaces removed).
        Callable items in the response list are invoked at match time.
        
        Returns:
            Tuple of (matched: bool, response: str)
            - matched: True if a pattern was found, False otherwise
            - response: The matched response, or empty string if no match
        """
        if not text:
            return (False, "")

        cleaned_text = "".join(text.split())

        for pattern, responses in self.patterns:
            if pattern in cleaned_text:
                chosen = random.choice(responses)
                # Support both static strings and handler callables
                response = chosen() if callable(chosen) else chosen
                return (True, response)

        # No match found
        return (False, "")