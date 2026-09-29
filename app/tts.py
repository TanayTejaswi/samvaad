import pyttsx3
import threading
import queue
import logging
import time

logger = logging.getLogger("samvaad.tts")

class TTSEngine:
    def __init__(self):
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        
        # Initialize pyttsx3 in the worker thread to avoid COM/ObjC threading issues
        self.engine = None 
        self.is_running = True
        self.thread.start()
        
    def _worker(self):
        # pyttsx3 MUST be initialized in the same thread it is run in!
        try:
            self.engine = pyttsx3.init()
            # Try to speed it up a bit if needed
            rate = self.engine.getProperty('rate')
            self.engine.setProperty('rate', rate + 20)
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            return
            
        while self.is_running:
            try:
                # Block until text is available
                text = self.q.get(timeout=1.0)
                
                logger.info(f"Speaking: '{text}'")
                self.engine.say(text)
                self.engine.runAndWait()
                
                self.q.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS Error during speech: {e}")
                
    def speak(self, text: str):
        """Queues text to be spoken asynchronously."""
        if text and str(text).strip():
            self.q.put(text)
            
    def shutdown(self):
        self.is_running = False
        self.thread.join()
