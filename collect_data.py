import cv2
import numpy as np
import os
import mediapipe as mp
import tkinter as tk
from tkinter import simpledialog

# Configurar MediaPipe para manos con mejor precisión
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7,  # Mayor precisión
    min_tracking_confidence=0.7    # Mejor seguimiento
)
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

# === FUNCIÓN MEJORADA PARA EXTRAER COORDENADAS ===
def extract_best_hand_landmarks(multi_hand_landmarks, handedness_results):
    """Extrae coordenadas de la mejor mano detectada de manera consistente"""
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
        return hands_data[0]['coords']
    elif len(hands_data) == 2:
        # Usar la mano de mayor confianza de manera consistente
        best_hand = max(hands_data, key=lambda x: x['confidence'])
        return best_hand['coords']
    else:
        return hands_data[0]['coords']

# Configuración inicial mejorada
num_sequences = 40  # Más secuencias para mejor entrenamiento
sequence_length = 30  # Frames por secuencia
min_confidence = 0.7  # Confianza mínima para guardar frame
cap = cv2.VideoCapture(0)

print("Sistema de Recolección de Datos para Lenguaje de Señas")
print("Instrucciones:")
print("   - Haz cada seña de forma clara y consistente")
print("   - Mantén las manos visibles en todo momento") 
print("   - Cada seña se grabará en 40 secuencias de 30 frames")
print("   - ESC para cancelar secuencia actual")
print("   - Q para pasar a la siguiente seña")

while True:
    # Preguntar si quiere agregar una nueva seña o salir
    sign = capture_new_sign()
    if not sign:
        break

    # Crear directorio para cada seña
    sign_dir = os.path.join(DATA_DIR, sign)
    if not os.path.exists(sign_dir):
        os.makedirs(sign_dir)

    print(f'\nPreparando recolección para: "{sign}"')
    print('Posiciona tus manos y presiona "Q" para comenzar')

    # Fase de preparación
    ready = False
    while not ready:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Mostrar preview
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            num_hands = 0
            
        cv2.putText(frame, f'Preparado para: {sign.upper()}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f' Manos detectadas: {num_hands}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, 'Presiona Q para empezar a grabar', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, 'ESC para cancelar esta seña', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow('Recolección de Datos - SignLanguage', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            ready = True
        elif key == 27:  # ESC
            print(f"❌ Cancelada recolección para '{sign}'")
            break
    
    if not ready:
        continue

    for sequence in range(num_sequences):
        frame_data = []
        frames_captured = 0
        frames_skipped = 0
        
        print(f'\n🎬 Iniciando secuencia {sequence + 1}/{num_sequences} para "{sign}"')
        print('Mantén la seña estable y clara durante 3-4 segundos')
        
        while len(frame_data) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Voltear la imagen horizontalmente para vista tipo espejo
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convertir a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            # Extraer coordenadas usando la función mejorada
            hand_coords = extract_best_hand_landmarks(results.multi_hand_landmarks, results)
            
            # Verificar si hay datos válidos (no todos ceros)
            has_valid_data = any(coord != 0.0 for coord in hand_coords)
            
            if has_valid_data and results.multi_hand_landmarks:
                frame_data.append(hand_coords)
                frames_captured += 1
                status_color = (0, 255, 0)  # Verde para frame válido
                status_text = f"Frame {len(frame_data)}/{sequence_length}"
            else:
                frames_skipped += 1
                status_color = (0, 0, 255)  # Rojo para frame inválido
                status_text = f"Sin mano detectada ({frames_skipped} omitidos)"
            
            # Dibujar todas las manos detectadas
            num_hands = 0
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Mostrar información detallada
            cv2.putText(frame, f' Seña: {sign.upper()}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f'Secuencia: {sequence + 1}/{num_sequences}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f' Manos: {num_hands}', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f'Válidos: {frames_captured} | ❌ Omitidos: {frames_skipped}', (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, 'ESC=Cancelar secuencia | Q=Siguiente seña', (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow(' Recolección de Datos - SignLanguage', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC para cancelar secuencia
                print(f" Secuencia {sequence} cancelada")
                break
            elif key == ord('q'):  # Q para terminar esta seña
                break

        # Guardar los datos recolectados solo si tenemos suficientes frames
        if len(frame_data) >= sequence_length:
            # Asegurar que tenemos exactamente sequence_length frames
            frame_data = frame_data[:sequence_length]
            npy_path = os.path.join(sign_dir, f'seq_{sequence}.npy')
            np.save(npy_path, frame_data)
            print(f'Secuencia {sequence + 1} guardada: {frames_captured} frames válidos')
        else:
            print(f'Secuencia {sequence + 1} descartada: solo {len(frame_data)} frames válidos')

    print(f'\n🎉 Recolección completada para "{sign}": {num_sequences} secuencias')

cap.release()
cv2.destroyAllWindows()

