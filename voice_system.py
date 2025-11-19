# Sistema de voz OFFLINE para detección de señas
import time
import threading
import pyttsx3

class VoiceSystem:
    """Sistema de voz offline usando pyttsx3"""
    
    def __init__(self):
        self.last_spoken = None
        self.last_speak_time = 0
        self.is_speaking = False
        
        # Inicializar motor de voz offline
        try:
            self.engine = pyttsx3.init()
            
            # Configurar propiedades de la voz
            self.engine.setProperty('rate', 150)  # Velocidad de habla
            self.engine.setProperty('volume', 1.0)  # Volumen máximo
            
            # Intentar configurar voz en español
            voices = self.engine.getProperty('voices')
            spanish_voice = None
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'español' in voice.name.lower():
                    spanish_voice = voice.id
                    break
            
            if spanish_voice:
                self.engine.setProperty('voice', spanish_voice)
                print("[VOZ] Motor de voz en español configurado")
            else:
                print("[VOZ] Usando voz predeterminada del sistema")
            
            print("[VOZ] Sistema de voz OFFLINE inicializado correctamente")
            self.preload_words()
            
        except Exception as e:
            print(f"[ERROR] Error inicializando motor de voz: {e}")
            self.engine = None
        
    def preload_words(self):
        """Cargar palabras desde signs.json"""
        import json
        import os
        
        words = []
        if os.path.exists('signs.json'):
            try:
                with open('signs.json', 'r', encoding='utf-8') as f:
                    signs_data = json.load(f)
                    words = list(signs_data.values())
                print(f"[VOZ] Cargadas {len(words)} palabras: {', '.join(words)}")
            except Exception as e:
                print(f"[ERROR] Error leyendo signs.json: {e}")
        
        print("[VOZ] Sistema listo para sintetizar voz")
    
    def speak_sync(self, text):
        """Habla de forma síncrona usando pyttsx3"""
        if self.is_speaking or not self.engine:
            return False
        
        try:
            self.is_speaking = True
            
            # Limpiar el texto para mejor pronunciación
            clean_text = text.replace("_", " ")
            
            print(f"[VOZ] Reproduciendo: {clean_text}")
            
            # Reproducir con pyttsx3
            self.engine.say(clean_text)
            self.engine.runAndWait()
            
            print(f"[VOZ] Completado: {clean_text}")
            return True
                
        except Exception as e:
            print(f"[ERROR] Error reproduciendo '{text}': {e}")
            return False
        finally:
            self.is_speaking = False
    
    def speak_async(self, text):
        """Habla de forma asíncrona en hilo separado"""
        def speak_thread():
            self.speak_sync(text)
        
        if not self.is_speaking:
            thread = threading.Thread(target=speak_thread, daemon=True)
            thread.start()
    
    def can_speak(self, text, min_interval=3):
        """Verifica si puede hablar basado en tiempo y palabra anterior"""
        current_time = time.time()
        
        if text != self.last_spoken or (current_time - self.last_speak_time) > min_interval:
            return True
        return False
    
    def speak_if_ready(self, text, min_interval=3, async_mode=False):
        """Habla solo si es apropiado hacerlo"""
        if self.can_speak(text, min_interval) and not self.is_speaking:
            self.last_spoken = text
            self.last_speak_time = time.time()
            
            if async_mode:
                self.speak_async(text)
            else:
                self.speak_sync(text)
            return True
        return False
    
    def reload_signs(self):
        """Recarga las señas desde signs.json"""
        import json
        import os
        
        if os.path.exists('signs.json'):
            try:
                with open('signs.json', 'r', encoding='utf-8') as f:
                    signs_data = json.load(f)
                    new_words = list(signs_data.values())
                        
                print(f"[VOZ] Sistema actualizado con {len(new_words)} palabras")
                return True
            except Exception as e:
                print(f"[ERROR] Error recargando signs.json: {e}")
                return False
        return False
    
    def cleanup(self):
        """Limpia recursos del motor de voz"""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass

# Función de prueba
if __name__ == "__main__":
    print("[TEST] Probando sistema de voz offline...")
    
    voice_system = VoiceSystem()
    
    # Probar palabras
    test_words = ["hola", "adios", "como", "mal"]
    
    for word in test_words:
        print(f"\n[TEST] Probando: {word}")
        success = voice_system.speak_sync(word)
        if success:
            print(f"[OK] {word} - OK")
        else:
            print(f"[ERROR] {word} - ERROR")
        time.sleep(1)
    
    voice_system.cleanup()
    print("\n[TEST] Prueba completada")