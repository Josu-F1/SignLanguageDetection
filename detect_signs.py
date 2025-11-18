import cv2
import numpy as np
import mediapipe as mp
import os
import json
import time
from keras.models import load_model
from collections import deque
from voice_system import VoiceSystem

# === CONFIGURACIÓN MEJORADA - ALTA PRECISIÓN ===
SEQ_LEN = 30
FEATURES = 63  # Mantenemos 63 para compatibilidad con el modelo actual
CONFIDENCE_THRESHOLD = 0.75  # 75% para mayor precisión
REPEAT_INTERVAL = 6.0  # Más tiempo para evitar repeticiones
MIN_STABLE_FRAMES = 15  # Más frames para mayor estabilidad
PROCESSING_COOLDOWN = 3.0  # Tiempo de espera entre detecciones (segundos)
MIN_PREDICTION_HISTORY = 8  # Mínimo de predicciones para promediar

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
print("[VOZ] Inicializando sistema de voz mejorado...")
voice_system = VoiceSystem()

# Recargar señas dinámicamente si es necesario
print("[VOZ] Sincronizando con señas actuales...")
voice_system.reload_signs()

def speak(text):
    """Habla la palabra detectada usando el sistema mejorado"""
    try:
        # Intentar usar el nuevo sistema de voz
        success = voice_system.speak_sync(text)
        if success:
            print(f"[VOZ] Completada: {text}")
        else:
            print(f"[VOZ] Primera tentativa falló, reintentando para: {text}")
            # Segundo intento: forzar regeneración del audio
            if text in voice_system.audio_files:
                del voice_system.audio_files[text]  # Limpiar cache
            success = voice_system.speak_sync(text)
            if success:
                print(f"[VOZ] Completada en segundo intento: {text}")
            else:
                print(f"[ERROR] Falló completamente para: {text}")
    except Exception as e:
        print(f"[ERROR] Error en síntesis de voz: {e}")
        print(f"[VOZ] Intentando generar audio dinámicamente para '{text}'...")

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
last_detection_time = 0  # Tiempo de la última detección procesada
prediction_history = deque(maxlen=15)  # Historial más largo para mejor promedio
current_stable_sign = None
stable_count = 0
confidence_history = deque(maxlen=15)  # Historial de confianzas más largo
processing_blocked = False  # Flag para bloquear procesamiento durante cooldown

# === CONTROL DE DETECCIÓN ===
detection_active = False  # Iniciar con detección INACTIVA
frames_without_detection = 0
MAX_FRAMES_WITHOUT_DETECTION = 90  # 3 segundos a 30fps

print("\n[SISTEMA] RECONOCIMIENTO DE SEÑAS")
print("=" * 50)
print("[CONTROLES]")
print("   ENTER - Activar/Desactivar detección")
print("   ESPACIO - Forzar voz (si hay seña detectada)")
print("   Q - Salir del programa")
print("=" * 50)
print("[INFO] Presiona ENTER para comenzar la detección...")
print(f"[CONFIG] Configuración de precisión:")
print(f"   - Confianza mínima: {CONFIDENCE_THRESHOLD*100:.0f}%")
print(f"   - Estabilidad requerida: {MIN_STABLE_FRAMES} frames")
print(f"   - Cooldown entre detecciones: {PROCESSING_COOLDOWN}s")
print(f"   - Historial mínimo: {MIN_PREDICTION_HISTORY} predicciones")
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

    # Solo agregar a la secuencia si la detección está activa
    if detection_active:
        sequence.append(best_hand_coords)
    
    # === PROCESAMIENTO DE DETECCIÓN ===
    detected_sign = None
    confidence_level = 0.0
    
    # === SISTEMA DE COOLDOWN Y PROCESAMIENTO INTELIGENTE ===
    current_time = time.time()
    
    # Verificar si estamos en periodo de cooldown
    if processing_blocked and (current_time - last_detection_time) < PROCESSING_COOLDOWN:
        # Mostrar estado de cooldown
        remaining_time = PROCESSING_COOLDOWN - (current_time - last_detection_time)
        # === AREA DE COOLDOWN - POSICION INFERIOR ===
        cv2.putText(frame, f'PROCESANDO...', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,165,0), 2)
        cv2.putText(frame, f'Espera: {remaining_time:.1f}s', (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,165,0), 2)
        cv2.putText(frame, f'Ultima: {last_spoken or "Ninguna"}', (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    elif processing_blocked and (current_time - last_detection_time) >= PROCESSING_COOLDOWN:
        # Terminar cooldown
        processing_blocked = False
        print(f"[SISTEMA] Cooldown terminado - Listo para nueva detección")
    
    # Cuando hay suficientes frames Y la detección está activa Y no hay cooldown
    if detection_active and len(sequence) == SEQ_LEN and not processing_blocked:
        X = np.expand_dims(np.array(sequence), axis=0)  # (1,30,63) - Mejor mano
        prediction = model.predict(X, verbose=0)
        
        # Agregar predicción al historial
        prediction_history.append(prediction[0])
        
        # Requiere más historial para mayor precisión
        if len(prediction_history) >= MIN_PREDICTION_HISTORY:
            # Promediar las últimas predicciones para mayor estabilidad
            avg_prediction = np.mean(list(prediction_history), axis=0)
            idx = np.argmax(avg_prediction)
            confidence_level = avg_prediction[idx]
        else:
            # Si no hay suficiente historial, mostrar que está recopilando
            # === AREA DE ANALISIS ===
            cv2.putText(frame, f'ANALIZANDO...', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.putText(frame, f'Progreso: {len(prediction_history)}/{MIN_PREDICTION_HISTORY}', (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            idx = np.argmax(prediction)
            confidence_level = prediction[0][idx]
        
        # Validar que el índice esté dentro del rango válido
        if idx < len(sign_labels):
            detected_sign = sign_labels[idx]
            
            # Solo considerar predicciones con alta confianza
            if confidence_level > CONFIDENCE_THRESHOLD:
                # Sistema de estabilización de predicciones
                if detected_sign == current_stable_sign:
                    stable_count += 1
                else:
                    current_stable_sign = detected_sign
                    stable_count = 1
                
                frames_without_detection = 0  # Resetear contador
                
                # Mostrar predicción actual con información detallada
                if stable_count >= MIN_STABLE_FRAMES and confidence_level > 0.80:  # 80% - ALTA PRECISIÓN
                    color = (0, 255, 0)  # Verde para predicción MUY confiable
                    status = "¡DETECTADO!"
                elif stable_count >= MIN_STABLE_FRAMES and confidence_level > 0.65:  # 65% - BUENA PRECISIÓN
                    color = (0, 255, 255)  # Amarillo para estable pero no muy confiable
                    status = "Probable"
                elif stable_count >= (MIN_STABLE_FRAMES // 2):  # Mitad de estabilidad
                    color = (255, 165, 0)  # Naranja para en proceso
                    status = "Analizando"
                else:
                    color = (255, 255, 0)  # Azul para inestable
                    status = "Detectando"
                
                # === ÁREA PRINCIPAL - PARTE SUPERIOR ===
                cv2.putText(frame, f'{status}: {detected_sign.upper()}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f'Confianza: {confidence_level:.1%}', (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame, f'Estabilidad: {stable_count}/{MIN_STABLE_FRAMES}', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(frame, f'Datos: {len(prediction_history)}/{MIN_PREDICTION_HISTORY}', (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                
                # Debug: Mostrar información detallada
                current_time = time.time()
                time_passed = current_time - last_speak_time
                
                # Solo mostrar información cada 10 frames para reducir spam
                if stable_count % 10 == 0 or stable_count == MIN_STABLE_FRAMES:
                    print(f"[DETECCION] {detected_sign} - Confianza: {confidence_level:.2f}, Estable: {stable_count}/{MIN_STABLE_FRAMES}, Tiempo: {time_passed:.1f}s")
                
                # Usar el sistema de voz SOLO con alta precisión y estabilidad
                if stable_count >= MIN_STABLE_FRAMES and confidence_level > 0.75:  # 75% - ALTA PRECISIÓN
                    # El sistema de voz decide si debe hablar o no
                    if voice_system.speak_if_ready(detected_sign, min_interval=6, async_mode=False):  # Síncrono para mejor control
                        print(f"[DETECTADO] {detected_sign} (confianza: {confidence_level:.2f}, estabilidad: {stable_count})")
                        last_spoken = detected_sign
                        last_speak_time = current_time
                        last_detection_time = current_time 
                        processing_blocked = True  # Activar cooldown
                        
                        # Limpiar historiales para la próxima detección
                        prediction_history.clear()
                        sequence.clear()
                        stable_count = 0
                        current_stable_sign = None
                        
                        print(f"Iniciando cooldown de {PROCESSING_COOLDOWN}s para mayor precisión")
                    
            else:
                # Confianza baja - resetear contador e incrementar frames sin detección
                stable_count = 0
                current_stable_sign = None
                frames_without_detection += 1
                
                # === SEÑAL DÉBIL - AREA MEDIA ===
                cv2.putText(frame, f'SEÑAL DÉBIL', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
                cv2.putText(frame, f'Confianza: {confidence_level:.1%}', (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
        else:
            # Índice inválido - tratar como no reconocida
            frames_without_detection += 1
            cv2.putText(frame, f'NO RECONOCIDA', (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)
            print(f"[ADVERTENCIA] El modelo predijo el índice {idx} pero solo hay {len(sign_labels)} señas")
    
    # === MANEJO DE ESTADOS DE DETECCIÓN ===
    elif detection_active and len(sequence) == SEQ_LEN:
        # Detección activa pero sin señas reconocidas
        frames_without_detection += 1
        if frames_without_detection > MAX_FRAMES_WITHOUT_DETECTION:
            cv2.putText(frame, f'NO RECONOCIDA', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.putText(frame, f'Sin señas: {frames_without_detection//30}s', (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            cv2.putText(frame, f'DETECTANDO...', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,165,0), 2)
    
    elif detection_active and len(sequence) < SEQ_LEN:
        # Recopilando datos para detección
        cv2.putText(frame, f'RECOPILANDO...', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f'Datos: {len(sequence)}/{SEQ_LEN}', (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    else:
        # Detección INACTIVA
        cv2.putText(frame, f'SISTEMA INACTIVO', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128,128,128), 2)
        cv2.putText(frame, f'Presiona ENTER para activar', (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    
    # === INFORMACIÓN INFERIOR - BIEN SEPARADA ===
    h = frame.shape[0]  # Altura del frame
    
    # INFORMACIÓN COMPACTA EN LA PARTE INFERIOR
    cv2.putText(frame, f'Señas: {", ".join(sign_labels)}', (10, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
    cv2.putText(frame, f'Config: {CONFIDENCE_THRESHOLD*100:.0f}% | {MIN_STABLE_FRAMES}f | {PROCESSING_COOLDOWN}s', (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,200), 1)
    cv2.putText(frame, f'Última: {last_spoken or "Ninguna"}', (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.putText(frame, f'ENTER=On/Off | ESPACIO=Voz | Q=Salir', (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,200,100), 1)
    
    # === ESTADO COMPACTO (lado derecho) ===
    w = frame.shape[1]  # Ancho del frame
    
    # Estado simple
    status_text = "ON" if detection_active else "OFF"
    cv2.putText(frame, status_text, (w - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if detection_active else (0,0,255), 2)
    
    # Manos detectadas
    cv2.putText(frame, f'Manos: {num_hands_detected}', (w - 130, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    
    # Cooldown activo
    if processing_blocked:
        remaining = PROCESSING_COOLDOWN - (current_time - last_detection_time)
        cv2.putText(frame, f'{remaining:.1f}s', (w - 80, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,165,0), 1)

    # === MOSTRAR FRAME FINAL ===
    cv2.imshow('Reconocimiento de Senas con Voz', frame)

    # === CONTROLES Y INFORMACIÓN ADICIONAL ===
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 13:  # ENTER para activar/desactivar detección
        detection_active = not detection_active
        if detection_active:
            print(f"[SISTEMA] DETECCION ACTIVADA")
            sequence.clear()  # Limpiar secuencia al activar
            frames_without_detection = 0
            stable_count = 0
            current_stable_sign = None
        else:
            print(f"[SISTEMA] DETECCION DESACTIVADA")
            sequence.clear()  # Limpiar secuencia al desactivar
    elif key == ord(' '):  # Barra espaciadora para forzar voz
        if detection_active and current_stable_sign and stable_count >= MIN_STABLE_FRAMES:
            print(f"[VOZ] Forzando voz: {current_stable_sign}")
            voice_system.speak_if_ready(current_stable_sign, min_interval=0, async_mode=False)  # Sin intervalo mínimo
            last_spoken = current_stable_sign
            last_speak_time = time.time()
        else:
            print("[ADVERTENCIA] No hay seña estable para reproducir o detección inactiva")

cap.release()
cv2.destroyAllWindows()
hands.close()
voice_system.cleanup()
print("[SISTEMA] Cerrado correctamente")
