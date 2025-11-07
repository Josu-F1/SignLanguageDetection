import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Directorio de datos
DATA_DIR = 'data'

# Configuración
TARGET_FRAMES = 30
N_FEATURES = 63  # 21 puntos x 3 coordenadas por mano

def normalize_sequence(seq, target_frames=30):
    """Normaliza una secuencia a un número fijo de frames"""
    if len(seq) == 0:
        # Si no hay frames, crear una secuencia de ceros
        return np.zeros((target_frames, N_FEATURES))
        
    if len(seq) > target_frames:
        # Si hay más frames, tomar los primeros target_frames
        return seq[:target_frames]
    elif len(seq) < target_frames:
        # Si hay menos frames, repetir el último frame
        last_frame = seq[-1]
        padding = np.tile(last_frame, (target_frames - len(seq), 1))
        return np.vstack((seq, padding))
    return seq

# Cargar datos
sequences, labels = [], []
signs = [d for d in sorted(os.listdir(DATA_DIR)) if os.path.isdir(os.path.join(DATA_DIR, d))]

print("Señas encontradas para entrenar:")
for idx, sign in enumerate(signs):
    print(f"{idx + 1}. {sign}")

for sign_idx, sign in enumerate(signs):
    sign_dir = os.path.join(DATA_DIR, sign)
    for sequence_file in sorted(os.listdir(sign_dir)):
        if sequence_file.endswith('.npy'):
            # Cargar y normalizar la secuencia
            seq = np.load(os.path.join(sign_dir, sequence_file))
            if len(seq) > 0:  # Solo añadir si tiene frames
                normalized_seq = normalize_sequence(seq)
                sequences.append(normalized_seq)
                labels.append(sign_idx)

print(f"\nTotal de secuencias cargadas: {len(sequences)}")

# Convertir a numpy arrays
X = np.array(sequences)
print(f"Shape de los datos: {X.shape}")  # Debería ser (n_sequences, 30, 63)
y = to_categorical(labels).astype(int)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear el modelo
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(signs), activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Configurar TensorBoard para visualización
log_dir = 'logs'
tensorboard_callback = TensorBoard(log_dir=log_dir)

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
          callbacks=[tensorboard_callback])

# Guardar el modelo
model.save('sign_language_model.keras')
print("Modelo guardado como 'sign_language_model.keras'")

# Guardar las etiquetas de las señas en formato JSON
import json
signs_dict = {}
for idx, sign in enumerate(signs):
    signs_dict[str(idx)] = sign

with open('signs.json', 'w', encoding='utf-8') as f:
    json.dump(signs_dict, f, indent=2, ensure_ascii=False)

print("Archivo 'signs.json' actualizado con las señas:")
for idx, sign in enumerate(signs):
    print(f"  {idx}: {sign}")