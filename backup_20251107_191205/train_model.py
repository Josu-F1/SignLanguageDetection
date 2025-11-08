import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

# === CONFIGURACIÓN MEJORADA ===
DATA_DIR = 'data'
TARGET_FRAMES = 30
N_FEATURES = 63  # 21 puntos x 3 coordenadas por mano
MIN_SEQUENCES_PER_SIGN = 10  # Reducido para permitir más señas (era 20)

def normalize_sequence(seq, target_frames=30):
    """Normaliza una secuencia a un número fijo de frames con interpolación inteligente"""
    if len(seq) == 0:
        print("⚠️ Secuencia vacía encontrada, llenando con ceros")
        return np.zeros((target_frames, N_FEATURES))
    
    # Convertir a numpy array si no lo es
    seq = np.array(seq)
    
    # Verificar que cada frame tenga el número correcto de features
    if seq.shape[1] != N_FEATURES:
        print(f"⚠️ Secuencia con {seq.shape[1]} features, esperadas {N_FEATURES}")
        # Ajustar si es necesario
        if seq.shape[1] < N_FEATURES:
            # Rellenar con ceros si faltan features
            padding = np.zeros((seq.shape[0], N_FEATURES - seq.shape[1]))
            seq = np.hstack((seq, padding))
        else:
            # Truncar si hay demasiados features
            seq = seq[:, :N_FEATURES]
    
    if len(seq) == target_frames:
        return seq
    elif len(seq) > target_frames:
        # Submuestrear de manera uniforme
        indices = np.linspace(0, len(seq) - 1, target_frames, dtype=int)
        return seq[indices]
    else:
        # Interpolación para secuencias cortas
        from scipy.interpolate import interp1d
        try:
            x_old = np.linspace(0, 1, len(seq))
            x_new = np.linspace(0, 1, target_frames)
            
            # Interpolación para cada feature
            interpolated = np.zeros((target_frames, N_FEATURES))
            for feature_idx in range(N_FEATURES):
                f = interp1d(x_old, seq[:, feature_idx], kind='linear', fill_value='extrapolate')
                interpolated[:, feature_idx] = f(x_new)
            
            return interpolated
        except:
            # Fallback: repetir último frame
            last_frame = seq[-1]
            padding = np.tile(last_frame, (target_frames - len(seq), 1))
            return np.vstack((seq, padding))

def validate_sequence_quality(seq):
    """Valida la calidad de una secuencia con criterios más permisivos"""
    if len(seq) == 0:
        return False, "Secuencia vacía"
    
    # Verificar que no sea toda ceros
    if np.all(seq == 0):
        return False, "Secuencia solo con ceros"
    
    # Verificar variación mínima (movimiento) - criterio más relajado
    variance = np.var(seq, axis=0)
    mean_variance = np.mean(variance)
    
    # Relajar el umbral para aceptar más secuencias
    if mean_variance < 0.0001:  # Mucho más permisivo
        return False, "Muy poca variación en la secuencia"
    
    # Verificar que haya al menos algunos frames con datos válidos
    non_zero_frames = np.sum(np.any(seq != 0, axis=1))
    if non_zero_frames < len(seq) * 0.3:  # Al menos 30% de frames con datos
        return False, "Demasiados frames vacíos"
    
    return True, "Secuencia válida"

def load_and_preprocess_data():
    """Carga y preprocesa todos los datos con validación"""
    sequences, labels, sign_names = [], [], []
    signs = [d for d in sorted(os.listdir(DATA_DIR)) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    print("🔍 Analizando datos disponibles:")
    print("-" * 50)
    
    valid_signs = []
    for sign_idx, sign in enumerate(signs):
        sign_dir = os.path.join(DATA_DIR, sign)
        sequence_files = [f for f in sorted(os.listdir(sign_dir)) if f.endswith('.npy')]
        
        valid_sequences = 0
        invalid_sequences = 0
        
        for sequence_file in sequence_files:
            try:
                seq = np.load(os.path.join(sign_dir, sequence_file))
                is_valid, reason = validate_sequence_quality(seq)
                
                if is_valid:
                    normalized_seq = normalize_sequence(seq)
                    sequences.append(normalized_seq)
                    labels.append(sign_idx)
                    valid_sequences += 1
                else:
                    invalid_sequences += 1
                    print(f"  ⚠️ {sequence_file}: {reason}")
                    
            except Exception as e:
                print(f"  ❌ Error cargando {sequence_file}: {e}")
                invalid_sequences += 1
        
        print(f"📁 {sign}: {valid_sequences} válidas, {invalid_sequences} inválidas")
        
        if valid_sequences >= MIN_SEQUENCES_PER_SIGN:
            valid_signs.append((sign_idx, sign))
            sign_names.append(sign)
        else:
            print(f"  ⚠️ Insuficientes secuencias para '{sign}' (mín. {MIN_SEQUENCES_PER_SIGN})")
    
    return sequences, labels, valid_signs, sign_names

# === CARGAR Y PREPARAR DATOS ===
print("🚀 Iniciando entrenamiento del modelo de lenguaje de señas")
print("=" * 60)

# Cargar datos con validación mejorada
sequences, labels, valid_signs, sign_names = load_and_preprocess_data()

if len(sequences) == 0:
    print("❌ No se encontraron secuencias válidas para entrenar")
    exit(1)

print(f"\n📊 Resumen de datos:")
print(f"  - Total secuencias válidas: {len(sequences)}")
print(f"  - Señas para entrenar: {len(sign_names)}")
print(f"  - Señas: {', '.join(sign_names)}")

# Convertir a numpy arrays
X = np.array(sequences)
print(f"\n🔧 Shape de los datos de entrada: {X.shape}")

# Reindexar labels para que sean consecutivos
print(f"🔧 Reindexando labels...")
print(f"  - Secuencias cargadas: {len(sequences)}")  
print(f"  - Labels originales: {len(labels)}")

# Crear mapeo de señas válidas
valid_sign_names = [name for _, name in valid_signs]
label_mapping = {name: idx for idx, name in enumerate(valid_sign_names)}

print(f"  - Señas válidas: {valid_sign_names}")

# Procesar solo las secuencias de señas válidas
filtered_sequences = []
filtered_labels = []

sign_names_list = [d for d in sorted(os.listdir(DATA_DIR)) if os.path.isdir(os.path.join(DATA_DIR, d))]

for seq_idx, (seq, original_label) in enumerate(zip(sequences, labels)):
    # Obtener el nombre de la seña original
    if original_label < len(sign_names_list):
        sign_name = sign_names_list[original_label]
        
        # Solo incluir si la seña es válida (tiene suficientes datos)
        if sign_name in label_mapping:
            filtered_sequences.append(seq)
            filtered_labels.append(label_mapping[sign_name])

# Actualizar arrays con datos filtrados
X = np.array(filtered_sequences)
new_labels = filtered_labels
y = to_categorical(new_labels).astype(int)
num_classes = len(label_mapping)

print(f"  ✅ Datos finales:")
print(f"    - Secuencias: {len(X)}")
print(f"    - Labels: {len(new_labels)}")
print(f"    - Clases: {num_classes}")

print(f"🎯 Número de clases: {num_classes}")
for name, idx in label_mapping.items():
    count = sum(1 for label in new_labels if label == idx)
    print(f"  - {name}: {count} secuencias")

# === DIVIDIR DATOS Y CREAR MODELO ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=new_labels)

print(f"\n📚 División de datos:")
print(f"  - Entrenamiento: {X_train.shape[0]} secuencias")
print(f"  - Validación: {X_test.shape[0]} secuencias")

# Crear modelo mejorado con regularización
print("\n🏗️ Construyendo modelo...")
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(TARGET_FRAMES, N_FEATURES)),
    Dropout(0.2),
    BatchNormalization(),
    
    LSTM(128, return_sequences=True, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.2),
    
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'), 
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Configurar optimizador con learning rate adaptativo
optimizer = Adam(learning_rate=0.001, decay=1e-6)

# Compilar modelo
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

print(f"📋 Resumen del modelo:")
model.summary()

# Configurar callbacks mejorados
callbacks = [
    TensorBoard(log_dir='logs', histogram_freq=1),
    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

# Entrenar modelo
print(f"\n🎓 Iniciando entrenamiento...")
history = model.fit(
    X_train, y_train,
    epochs=100,  # Más épocas con early stopping
    batch_size=16,  # Batch size más pequeño para mejor convergencia
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# === EVALUACIÓN Y GUARDADO ===
print(f"\n🧪 Evaluando modelo en datos de prueba...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"📊 Precisión en datos de prueba: {test_accuracy:.4f}")
print(f"📊 Pérdida en datos de prueba: {test_loss:.4f}")

# Predicciones para reporte detallado
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Generar reporte de clasificación
class_names = [name for name, idx in sorted(label_mapping.items(), key=lambda x: x[1])]
print(f"\n📈 Reporte de clasificación:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Guardar el modelo
print(f"\n💾 Guardando modelo...")
model.save('sign_language_model.keras')
print("✅ Modelo guardado como 'sign_language_model.keras'")

# Crear y guardar mapping de señas
signs_dict = {}
for name, idx in label_mapping.items():
    signs_dict[str(idx)] = name

with open('signs.json', 'w', encoding='utf-8') as f:
    json.dump(signs_dict, f, indent=2, ensure_ascii=False)

print(f"\n📋 Archivo 'signs.json' actualizado con {len(signs_dict)} señas:")
for idx, sign in sorted(signs_dict.items(), key=lambda x: int(x[0])):
    print(f"  {idx}: {sign}")

# Guardar estadísticas del entrenamiento
training_stats = {
    'num_classes': num_classes,
    'total_sequences': len(sequences),
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'signs': signs_dict,
    'model_params': {
        'target_frames': TARGET_FRAMES,
        'n_features': N_FEATURES,
        'epochs_trained': len(history.history['loss']),
        'batch_size': 16
    }
}

with open('training_stats.json', 'w', encoding='utf-8') as f:
    json.dump(training_stats, f, indent=2, ensure_ascii=False)

print(f"\n🎉 ¡Entrenamiento completado exitosamente!")
print(f"📊 Precisión final: {test_accuracy:.2%}")
print(f"🎯 Modelo listo para detectar {num_classes} señas diferentes")
print("=" * 60)