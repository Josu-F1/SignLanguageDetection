# Sistema de voz alternativo para detección de señas
import os
import time
import tempfile
import threading
from gtts import gTTS
import pygame

class VoiceSystem:
    """Sistema de voz robusto con múltiples opciones"""
    
    def __init__(self):
        self.audio_files = {}  # Cache de archivos de audio
        self.last_spoken = None
        self.last_speak_time = 0
        self.is_speaking = False
        
        # Inicializar pygame mixer
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Pre-generar archivos de audio para las palabras comunes
        self.preload_words()
        
    def preload_words(self):
        """Pre-genera archivos de audio cargando desde signs.json"""
        import json
        import os
        
        # Cargar palabras desde signs.json si existe
        words = []
        if os.path.exists('signs.json'):
            try:
                with open('signs.json', 'r', encoding='utf-8') as f:
                    signs_data = json.load(f)
                    words = list(signs_data.values())
                print(f"📋 Cargadas {len(words)} palabras desde signs.json: {', '.join(words)}")
            except Exception as e:
                print(f"⚠️ Error leyendo signs.json: {e}")
                words = ["hola", "adios", "como_estas", "mal", "como", "cuanto", "sientes"]
        else:
            # Palabras por defecto si no existe signs.json
            words = ["hola", "adios", "como_estas", "mal", "como", "cuanto", "sientes"]
            print("📋 Usando palabras por defecto (signs.json no encontrado)")
        
        print("🎵 Pre-cargando archivos de voz...")
        for word in words:
            try:
                self.generate_audio_file(word)
                print(f"✅ Audio generado para: {word}")
            except Exception as e:
                print(f"⚠️ Error generando audio para {word}: {e}")
        print("🎵 Pre-carga de audio completada")
    
    def generate_audio_file(self, text):
        """Genera un archivo de audio para el texto dado"""
        if text in self.audio_files:
            return self.audio_files[text]
        
        try:
            # Crear archivo temporal
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_path = temp_file.name
            temp_file.close()
            
            # Generar audio con gTTS
            tts = gTTS(text=text, lang='es', slow=False)
            tts.save(temp_path)
            
            # Guardar en cache
            self.audio_files[text] = temp_path
            return temp_path
            
        except Exception as e:
            print(f"❌ Error generando audio para '{text}': {e}")
            return None
    
    def speak_sync(self, text):
        """Habla de forma síncrona, generando audio dinámicamente si es necesario"""
        if self.is_speaking:
            return False
        
        try:
            self.is_speaking = True
            
            # Obtener archivo de audio (generar dinámicamente si no existe)
            audio_file = self.audio_files.get(text)
            if not audio_file:
                print(f"🎵 Generando audio dinámicamente para: '{text}'")
                audio_file = self.generate_audio_file(text)
                if audio_file:
                    print(f"✅ Audio generado exitosamente para: '{text}'")
            
            if audio_file and os.path.exists(audio_file):
                print(f"🔊 Reproduciendo: {text}")
                
                # Reproducir con pygame
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                # Esperar a que termine la reproducción
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                print(f"✅ Completado: {text}")
                return True
            else:
                print(f"❌ No se pudo generar audio para: {text}")
                return False
                
        except Exception as e:
            print(f"❌ Error reproduciendo '{text}': {e}")
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
        """Recarga las señas desde signs.json y pre-genera audio si es necesario"""
        import json
        import os
        
        if os.path.exists('signs.json'):
            try:
                with open('signs.json', 'r', encoding='utf-8') as f:
                    signs_data = json.load(f)
                    new_words = list(signs_data.values())
                
                # Generar audio para palabras nuevas que no estén en cache
                for word in new_words:
                    if word not in self.audio_files:
                        print(f"🆕 Nueva palabra detectada: '{word}' - Generando audio...")
                        self.generate_audio_file(word)
                        
                print(f"🔄 Sistema actualizado con {len(new_words)} palabras")
                return True
            except Exception as e:
                print(f"❌ Error recargando signs.json: {e}")
                return False
        return False
    
    def cleanup(self):
        """Limpia recursos y archivos temporales"""
        pygame.mixer.quit()
        for audio_file in self.audio_files.values():
            try:
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
            except:
                pass

# Función de prueba
if __name__ == "__main__":
    print("🧪 Probando sistema de voz alternativo...")
    
    voice_system = VoiceSystem()
    
    # Probar palabras
    test_words = ["hola", "adios", "como", "mal"]
    
    for word in test_words:
        print(f"\n📢 Probando: {word}")
        success = voice_system.speak_sync(word)
        if success:
            print(f"✅ {word} - OK")
        else:
            print(f"❌ {word} - ERROR")
        time.sleep(1)
    
    voice_system.cleanup()
    print("\n🏁 Prueba completada")