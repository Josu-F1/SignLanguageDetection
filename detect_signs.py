import cv2
import numpy as np
import mediapipe as mp
import os
import json
import pyttsx3
import time
from keras.models import load_model
from collections import deque

# === CONFIGURACIÓN ===
SEQ_LEN = 30
FEATURES = 63  # tu modelo fue entrenado con una sola mano
CONFIDENCE_THRESHOLD = 0.5
REPEAT_INTERVAL = 1.0  # segundos entre repeticiones de la misma palabra

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
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === CONFIGURAR VOZ ===
engine = pyttsx3.init()
engine.setProperty('rate', 190)
voices = engine.getProperty('voices')
for v in voices:
    if 'spanish' in v.name.lower():
        engine.setProperty('voice', v.id)
        break

def speak(text):
    """Habla la palabra detectada"""
    engine.say(text)
    engine.runAndWait()

# === FUNCIÓN PARA EXTRAER COORDENADAS ===
def extract_hand_landmarks(hand_landmarks):
    if not hand_landmarks:
        return [0.0] * 63
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

# === CAPTURA DE VIDEO ===
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=SEQ_LEN)
last_spoken = None
last_speak_time = 0

print("Presiona 'q' para salir.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar manos
    results_hands = hands.process(rgb)
    right_hand = [0.0] * 63

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            right_hand = extract_hand_landmarks(hand_landmarks)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    sequence.append(right_hand)

    # Cuando hay suficientes frames
    if len(sequence) == SEQ_LEN:
        X = np.expand_dims(np.array(sequence), axis=0)  # (1,30,63)
        prediction = model.predict(X, verbose=0)
        idx = np.argmax(prediction)
        
        # Validar que el índice esté dentro del rango válido
        if idx < len(sign_labels):
            sign = sign_labels[idx]
            confidence = prediction[0][idx]
            
            cv2.putText(frame, f'Seña: {sign}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f'Confianza: {confidence:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        else:
            cv2.putText(frame, f'Error: Índice inválido {idx}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            print(f"Error: El modelo predijo el índice {idx} pero solo hay {len(sign_labels)} señas")

        current_time = time.time()
        if confidence > CONFIDENCE_THRESHOLD:
            # Hablar solo si cambia la palabra o pasa el intervalo de 1 segundo
            if sign != last_spoken or (current_time - last_speak_time > REPEAT_INTERVAL):
                speak(sign)
                last_spoken = sign
                last_speak_time = current_time

    cv2.imshow('Reconocimiento de Señas (Solo Manos)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
