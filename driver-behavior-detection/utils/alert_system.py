import threading
import time
import os
import sys

# Try to import pyttsx3, if not available, use print as fallback
try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False
    print("Warning: pyttsx3 not available, using console output for alerts")

class AlertSystem:
    def __init__(self):
        self.engine = None
        if HAS_TTS:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.9)

                voices = self.engine.getProperty('voices')
                if voices:
                    self.engine.setProperty('voice', voices[0].id)
            except:
                print("Warning: Text-to-speech initialization failed")
                self.engine = None

        self.last_alert_time = {}
        self.alert_cooldown = 3

        self.alert_messages = {
            'DROWSY': 'Cảnh báo: Bạn đang buồn ngủ! Hãy dừng xe nghỉ ngơi!',
            'YAWNING': 'Cảnh báo: Bạn đang ngáp nhiều! Cần tập trung lái xe!',
            'DISTRACTED': 'Cảnh báo: Bạn đang mất tập trung! Hãy nhìn thẳng về phía trước!',
            'PHONE USAGE': 'Cảnh báo: Không sử dụng điện thoại khi lái xe!',
            'SMOKING DETECTED': 'Cảnh báo: Không hút thuốc khi lái xe!',
            'GAZE DISTRACTED': 'Cảnh báo: Hãy tập trung nhìn đường!'
        }

        self.is_speaking = False

    def play_beep(self, frequency=1000, duration=0.2):
        # Simple console beep
        if sys.platform == "darwin":  # macOS
            os.system("printf '\a'")  # Terminal beep
        elif sys.platform.startswith("win"):  # Windows
            import winsound
            winsound.Beep(frequency, int(duration * 1000))
        else:  # Linux
            os.system("printf '\a'")  # Terminal beep

    def speak_alert(self, message):
        if not self.is_speaking:
            self.is_speaking = True
            def speak():
                try:
                    if self.engine:
                        self.engine.say(message)
                        self.engine.runAndWait()
                    else:
                        # Fallback to console output
                        print(f"\n🔊 ALERT: {message}\n")
                except:
                    print(f"\n🔊 ALERT: {message}\n")
                self.is_speaking = False

            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()

    def trigger_alert(self, alert_type, custom_message=None):
        current_time = time.time()

        if alert_type in self.last_alert_time:
            if current_time - self.last_alert_time[alert_type] < self.alert_cooldown:
                return

        self.last_alert_time[alert_type] = current_time

        if alert_type in ['DROWSY', 'PHONE USAGE', 'SMOKING DETECTED']:
            for _ in range(3):
                self.play_beep(1500, 0.1)
                time.sleep(0.1)
        else:
            self.play_beep(1000, 0.3)

        message = custom_message or self.alert_messages.get(alert_type, 'Cảnh báo!')
        self.speak_alert(message)

        return message

    def get_alert_level(self, alert_type):
        levels = {
            'DROWSY': 'CRITICAL',
            'PHONE USAGE': 'HIGH',
            'SMOKING DETECTED': 'HIGH',
            'DISTRACTED': 'MEDIUM',
            'YAWNING': 'LOW',
            'GAZE DISTRACTED': 'MEDIUM'
        }
        return levels.get(alert_type, 'LOW')

    def get_alert_color(self, alert_type):
        level = self.get_alert_level(alert_type)
        colors = {
            'CRITICAL': (255, 0, 0),
            'HIGH': (255, 100, 0),
            'MEDIUM': (255, 200, 0),
            'LOW': (255, 255, 0)
        }
        return colors.get(level, (255, 255, 255))

    def cleanup(self):
        try:
            if self.engine:
                self.engine.stop()
        except:
            pass