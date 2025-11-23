# Sistema de voz con Google TTS (gTTS) - Requiere conexion a internet
import os
import time
import tempfile
import threading
from gtts import gTTS
import pygame
import json # Asegurarse de que json esté importado aquí

class VoiceSystem:
    """Sistema de voz usando Google TTS (gTTS) y pygame"""
    
    def __init__(self):
        self.last_spoken = None
        self.last_speak_time = 0
        self.is_speaking = False
        self.audio_cache = {}
        
        # Inicializar pygame mixer
        try:
            pygame.mixer.init()
            print("[VOZ] Sistema de voz ONLINE inicializado (gTTS + pygame)")
            self.preload_words()
        except Exception as e:
            print(f"[ERROR] Error inicializando pygame: {e}")
        
    def preload_words(self):
        """Pre-genera archivos de audio para las palabras en signs.json"""
        # Ya no necesitamos importar json aquí si está arriba
        
        words = []
        if os.path.exists('signs.json'):
            try:
                with open('signs.json', 'r', encoding='utf-8') as f:
                    signs_data = json.load(f)
                    words = list(signs_data.values())
            except Exception as e:
                print(f"[ERROR] Error leyendo signs.json: {e}")
                return
        
        print(f"[VOZ] Pre-generando audio para {len(words)} palabras...")
        
        for word in words:
            try:
                # Generar audio con el texto limpio
                clean_word = word.replace("_", " ")
                tts = gTTS(text=clean_word, lang='es', slow=False)
                
                # Guardar en archivo temporal
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_file.close()
                tts.save(temp_file.name)
                
                # Guardar en cache, usando la clave original (con guion bajo)
                self.audio_cache[word] = temp_file.name
                
            except Exception as e:
                print(f"[ERROR] Error generando audio para '{word}': {e}")
        
        print(f"[VOZ] Audio pre-generado: {len(self.audio_cache)} archivos listos")
    
    def speak_sync(self, text):
        """Habla de forma sincrona. 'text' puede contener guiones bajos."""
        if self.is_speaking:
            return False
        
        try:
            self.is_speaking = True
            
            # 🔑 LIMPIEZA CLAVE 1: Obtener la versión limpia para TTS y logs
            clean_text = text.replace("_", " ")
            
            # Verificar si ya existe en cache (usa la clave original con guion bajo)
            if text in self.audio_cache:
                audio_file = self.audio_cache[text]
            else:
                # Generar nuevo audio usando el texto limpio
                tts = gTTS(text=clean_text, lang='es', slow=False)
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_file.close()
                tts.save(temp_file.name)
                
                audio_file = temp_file.name
                self.audio_cache[text] = audio_file
            
            # Reproducir audio
            print(f"[VOZ] Reproduciendo: {clean_text}")
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Esperar a que termine
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            print(f"[VOZ] Completado: {clean_text}")
            return True
                
        except Exception as e:
            print(f"[ERROR] Error reproduciendo '{text}': {e}")
            return False
        finally:
            self.is_speaking = False
    
    def speak_async(self, text):
        """Habla de forma asincrona en hilo separado"""
        def speak_thread():
            self.speak_sync(text)
        
        if not self.is_speaking:
            thread = threading.Thread(target=speak_thread, daemon=True)
            thread.start()
    
    def can_speak(self, text, min_interval=3):
        """Verifica si puede hablar basado en tiempo y palabra anterior"""
        current_time = time.time()
        
        # 🔑 LIMPIEZA CLAVE 2: Limpiar para la lógica de comparación
        clean_text = text.replace("_", " ")
        
        if clean_text != self.last_spoken or (current_time - self.last_speak_time) > min_interval:
            return True
        return False
    
    def speak_if_ready(self, text, min_interval=3, async_mode=False):
        """Habla solo si es apropiado hacerlo"""
        
        # 🔑 LIMPIEZA CLAVE 3: Limpiar para el almacenamiento de última palabra
        clean_text = text.replace("_", " ")
        
        if self.can_speak(text, min_interval) and not self.is_speaking:
            self.last_spoken = clean_text # <--- ALMACENAR LA VERSIÓN LIMPIA
            self.last_speak_time = time.time()
            
            if async_mode:
                self.speak_async(text)
            else:
                self.speak_sync(text)
            return True
        return False
    
    def reload_signs(self):
        """Recarga las senas desde signs.json y regenera audios"""
        
        if os.path.exists('signs.json'):
            try:
                with open('signs.json', 'r', encoding='utf-8') as f:
                    signs_data = json.load(f)
                    new_words = list(signs_data.values())
                
                # Generar audio para nuevas palabras
                for word in new_words:
                    if word not in self.audio_cache:
                        try:
                            # Generar audio con el texto limpio
                            clean_word = word.replace("_", " ")
                            tts = gTTS(text=clean_word, lang='es', slow=False)
                            
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                            temp_file.close()
                            tts.save(temp_file.name)
                            
                            self.audio_cache[word] = temp_file.name
                        except Exception as e:
                            print(f"[ERROR] Error generando audio para '{word}': {e}")
                
                print(f"[VOZ] Sistema actualizado con {len(new_words)} palabras")
                return True
            except Exception as e:
                print(f"[ERROR] Error recargando signs.json: {e}")
                return False
        return False
    
    def cleanup(self):
        """Limpia archivos temporales y recursos"""
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass
        
        # Eliminar archivos de cache
        for audio_file in self.audio_cache.values():
            try:
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
            except:
                pass

# Funcion de prueba
if __name__ == "__main__":
    print("[TEST] Probando sistema de voz online...")
    
    # Simular una sena con guion bajo
    test_words = ["hola", "como_estas", "adios", "mal"] 
    
    voice_system = VoiceSystem()
    
    for word in test_words:
        print(f"\n[TEST] Probando: {word}")
        success = voice_system.speak_sync(word)
        if success:
            print(f"[OK] {word} - OK")
        else:
            print(f"[ERROR] {word} - ERROR")
        time.sleep(1)
        
    # Prueba de repetición (debería hablar solo la primera vez)
    print("\n[TEST] Probando repetición rápida (solo debe hablar una vez)")
    voice_system.speak_if_ready("como_estas", min_interval=5, async_mode=False)
    voice_system.speak_if_ready("como_estas", min_interval=5, async_mode=False) # No debería hablar
    time.sleep(6)
    voice_system.speak_if_ready("como_estas", min_interval=5, async_mode=False) # Debería hablar ahora
    
    voice_system.cleanup()
    print("\n[TEST] Prueba completada")