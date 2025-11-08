import cv2
import numpy as np
import mediapipe as mp
import os
import json
import time
from keras.models import load_model
from collections import deque
from voice_system import VoiceSystem

# === CONFIGURACIÓN MEJORADA ===
SEQ_LEN = 30
FEATURES = 63  # Mantenemos 63 para compatibilidad con el modelo actual
CONFIDENCE_THRESHOLD = 0.50  # 50% para detectar MUY fácilmente
REPEAT_INTERVAL = 4.0  # Más tiempo para evitar repeticiones
MIN_STABLE_FRAMES = 8  # MÁS FRAMES para mayor estabilidad

# === CARGAR MODELO Y SEÑAS ===
model = load_model('sign_language_model.keras')

# Información del modelo
print("Modelo cargado exitosamente")
try:
    # Crear un input de prueba para obtener la forma de salida
    test_input = np.zeros((1, SEQ_LEN, FEATURES))
    test_output = model.predict(test_input, verbose=0)
    print(f"Número de clases del modelo: {test_output.shape[-1]}")
except Exception as e:
    print(f"No se pudo determinar el número de clases: {e}")

if os.path.exists('signs.json'):
    with open('signs.json', 'r', encoding='utf-8') as f:
        signs = json.load(f)
else:
    raise FileNotFoundError("No se encontró 'signs.json' con las etiquetas de las señas.")

sign_labels = [signs[k] for k in sorted(signs.keys(), key=lambda x: int(x))]
print("Señas disponibles:", sign_labels)
print(f"Número de señas en signs.json: {len(sign_labels)}")

# === CONFIGURAR MEDIAPIPE ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # DETECTAR LAS DOS MANOS
    min_detection_confidence=0.5,  # Reducido para mejor detección
    min_tracking_confidence=0.5   # Reducido para mejor detección
)

# === CONFIGURAR VOZ - SISTEMA MEJORADO ===
print("🔊 Inicializando sistema de voz mejorado...")
voice_system = VoiceSystem()

# Recargar señas dinámicamente si es necesario
print("🔄 Sincronizando con señas actuales...")
voice_system.reload_signs()

def speak(text):
    """Habla la palabra detectada usando el sistema mejorado"""
    try:
        # Intentar usar el nuevo sistema de voz
        success = voice_system.speak_sync(text)
        if success:
            print(f"✅ Voz completada: {text}")
        else:
            print(f"⚠️ Primera tentativa falló, reintentando para: {text}")
            # Segundo intento: forzar regeneración del audio
            if text in voice_system.audio_files:
                del voice_system.audio_files[text]  # Limpiar cache
            success = voice_system.speak_sync(text)
            if success:
                print(f"✅ Voz completada en segundo intento: {text}")
            else:
                print(f"❌ Falló completamente para: {text}")
    except Exception as e:
        print(f"❌ Error en síntesis de voz: {e}")
        print(f"🔄 Intentando generar audio dinámicamente para '{text}'...")

# === FUNCIÓN PARA EXTRAER COORDENADAS DE DOS MANOS ===
def extract_best_hand_landmarks(multi_hand_landmarks, handedness_results):
    """Extrae coordenadas de la mejor mano detectada o combina ambas inteligentemente"""
    if not multi_hand_landmarks:
        return [0.0] * 63
    
    hands_data = []
    
    # Recopilar información de todas las manos
    if handedness_results and handedness_results.multi_handedness:
        for hand_landmarks, handedness in zip(multi_hand_landmarks, handedness_results.multi_handedness):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            
            confidence = handedness.classification[0].score
            hand_label = handedness.classification[0].label
            
            hands_data.append({
                'coords': coords,
                'confidence': confidence,
                'label': hand_label
            })
    else:
        # Si no hay información de handedness, usar la primera mano
        coords = []
        for lm in multi_hand_landmarks[0].landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return coords
    
    if len(hands_data) == 1:
        # Solo una mano detectada
        return hands_data[0]['coords']
    elif len(hands_data) == 2:
        # Dos manos detectadas - usar la de mayor confianza
        best_hand = max(hands_data, key=lambda x: x['confidence'])
        return best_hand['coords']
    else:
        # Más de 2 manos (raro) - usar la primera
        return hands_data[0]['coords']

# === CAPTURA DE VIDEO ===
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=SEQ_LEN)

# Variables para control de voz y estabilidad
last_spoken = None
last_speak_time = 0
prediction_history = deque(maxlen=10)  # Historial de predicciones para promediar
current_stable_sign = None
stable_count = 0
confidence_history = deque(maxlen=10)  # Historial de confianzas

print("Presiona 'q' para salir.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar ambas manos
    results_hands = hands.process(rgb)
    
    # Extraer coordenadas de la mejor mano
    best_hand_coords = extract_best_hand_landmarks(results_hands.multi_hand_landmarks, results_hands)
    
    # Dibujar todas las manos detectadas
    num_hands_detected = 0
    if results_hands.multi_hand_landmarks:
        num_hands_detected = len(results_hands.multi_hand_landmarks)
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    sequence.append(best_hand_coords)

    # Cuando hay suficientes frames
    if len(sequence) == SEQ_LEN:
        X = np.expand_dims(np.array(sequence), axis=0)  # (1,30,63) - Mejor mano
        prediction = model.predict(X, verbose=0)
        
        # Agregar predicción al historial
        prediction_history.append(prediction[0])
        
        # Si tenemos suficiente historial, promediar las predicciones
        if len(prediction_history) >= 5:
            # Promediar las últimas predicciones para mayor estabilidad
            avg_prediction = np.mean(list(prediction_history), axis=0)
            idx = np.argmax(avg_prediction)
            confidence = avg_prediction[idx]
        else:
            idx = np.argmax(prediction)
            confidence = prediction[0][idx]
        
        # Validar que el índice esté dentro del rango válido
        if idx < len(sign_labels):
            sign = sign_labels[idx]
            
            # Solo considerar predicciones con alta confianza
            if confidence > CONFIDENCE_THRESHOLD:
                # Sistema de estabilización de predicciones
                if sign == current_stable_sign:
                    stable_count += 1
                else:
                    current_stable_sign = sign
                    stable_count = 1
                
                # Mostrar predicción actual con más información
                if stable_count >= MIN_STABLE_FRAMES and confidence > 0.50:  # 50% - MUY FÁCIL
                    color = (0, 255, 0)  # Verde para predicción MUY confiable
                    status = "¡DETECTADO!"
                elif stable_count >= MIN_STABLE_FRAMES:
                    color = (0, 255, 255)  # Amarillo para estable pero no muy confiable
                    status = "Estable"
                else:
                    color = (255, 255, 0)  # Azul para inestable
                    status = "Procesando"
                
                cv2.putText(frame, f'{status}: {sign.upper()}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                cv2.putText(frame, f'Confianza: {confidence:.3f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.putText(frame, f'Estabilidad: {stable_count}/{MIN_STABLE_FRAMES}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f'Predicciones: {len(prediction_history)}/10', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                
                # Debug: Mostrar información detallada
                current_time = time.time()
                time_passed = current_time - last_speak_time
                
                # Solo mostrar información cada 10 frames para reducir spam
                if stable_count % 10 == 0 or stable_count == MIN_STABLE_FRAMES:
                    print(f"📊 {sign} - Confianza: {confidence:.2f}, Estable: {stable_count}/{MIN_STABLE_FRAMES}, Tiempo: {time_passed:.1f}s")
                
                # Usar el sistema de voz inteligente SOLO si está muy estable
                if stable_count >= MIN_STABLE_FRAMES and confidence > 0.50:  # 50% - MUY FÁCIL
                    # El sistema de voz decide si debe hablar o no
                    if voice_system.speak_if_ready(sign, min_interval=4, async_mode=False):  # Síncrono para mejor control
                        print(f"🗣️ ¡DETECTADO Y HABLANDO!: {sign} (confianza: {confidence:.2f}, estabilidad: {stable_count})")
                        last_spoken = sign
                        last_speak_time = current_time
                    
            else:
                # Confianza baja - resetear contador
                stable_count = 0
                current_stable_sign = None
                cv2.putText(frame, f'Detectando...', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
                cv2.putText(frame, f'Confianza: {confidence:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
        else:
            cv2.putText(frame, f'Error: Índice inválido {idx}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            print(f"Error: El modelo predijo el índice {idx} pero solo hay {len(sign_labels)} señas")
    
    else:
        # No hay suficientes frames aún
        cv2.putText(frame, f'Recopilando datos... ({len(sequence)}/{SEQ_LEN})', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # Mostrar información de detección de manos
    cv2.putText(frame, f'Manos detectadas: {num_hands_detected}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Mostrar información adicional y ayuda
    cv2.putText(frame, f'Señas: {", ".join(sign_labels)}', (10, frame.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(frame, f'Umbral confianza: {CONFIDENCE_THRESHOLD}', (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.putText(frame, f'Ultima palabra: {last_spoken or "Ninguna"}', (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.putText(frame, f'CONSEJOS:', (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(frame, f'- Haz la seña lentamente y mantenla 2-3 segundos', (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    cv2.putText(frame, f'- Q=Salir | ESPACIO=Forzar voz', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    
    cv2.imshow('🤟 Reconocimiento de Señas con Voz 🔊', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Barra espaciadora para forzar voz
        if current_stable_sign and stable_count >= MIN_STABLE_FRAMES:
            print(f"🔊 Forzando voz: {current_stable_sign}")
            speak(current_stable_sign)
            last_spoken = current_stable_sign
            last_speak_time = time.time()

cap.release()
cv2.destroyAllWindows()
hands.close()
voice_system.cleanup()
print("🏁 Sistema cerrado correctamente")
