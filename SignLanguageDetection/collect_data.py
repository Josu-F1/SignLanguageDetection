import cv2
import numpy as np
import os
import mediapipe as mp
import tkinter as tk
from tkinter import simpledialog

# Configurar MediaPipe para manos y cara
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Inicializar Tkinter para diálogos
root = tk.Tk()
root.withdraw()  # Ocultar la ventana principal

# Crear directorio para guardar los datos
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Función para capturar nueva seña
def capture_new_sign():
    # Pedir el nombre de la seña
    sign = simpledialog.askstring("Nueva Seña", "¿Qué seña vas a realizar?")
    if not sign:
        return None
    # Limpiar el nombre de la seña
    sign = sign.lower().replace(" ", "_")
    return sign

# Configuración inicial
num_sequences = 30
cap = cv2.VideoCapture(0)

while True:
    # Preguntar si quiere agregar una nueva seña o salir
    sign = capture_new_sign()
    if not sign:
        break

    # Crear directorio para cada seña
    sign_dir = os.path.join(DATA_DIR, sign)
    if not os.path.exists(sign_dir):
        os.makedirs(sign_dir)

    print(f'Recolectando datos para la seña: {sign}')
    print('Presiona "Q" para empezar a grabar la seña')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, f'Preparado para capturar: {sign}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for sequence in range(num_sequences):
        frame_data = []
        while len(frame_data) < 30:
            # Capturar 30 frames por secuencia
            ret, frame = cap.read()
            if not ret:
                continue
            # Voltear la imagen horizontalmente para una vista tipo espejo
            frame = cv2.flip(frame, 1)
            # Convertir a RGB para MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar los puntos de referencia
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # Recolectar coordenadas
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    frame_data.append(landmarks)

            cv2.putText(frame, f'Grabando {sign} - Secuencia {sequence}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

        # Guardar los datos recolectados
        if frame_data:
            npy_path = os.path.join(sign_dir, f'seq_{sequence}.npy')
            np.save(npy_path, frame_data)
            print(f'Guardada secuencia {sequence} para {sign}')

cap.release()
cv2.destroyAllWindows()

