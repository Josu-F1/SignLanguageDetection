# Sistema de Reconocimiento de Lenguaje de Señas en Tiempo Real

## Descripción General

Sistema avanzado de reconocimiento automático de lenguaje de señas que utiliza redes neuronales LSTM y visión por computadora para detectar y traducir señas en tiempo real, con síntesis de voz integrada para retroalimentación auditiva inmediata.

## Características Principales

- ✅ **Detección en Tiempo Real**: Procesamiento a 30 fps con visualización inmediata
- ✅ **Alta Precisión**: Umbral de confianza del 66% con sistema de estabilización de 15 frames
- ✅ **Síntesis de Voz Dual**: Modo offline (pyttsx3) y online (gTTS)
- ✅ **Detección Bilateral**: Soporta ambas manos con selección inteligente
- ✅ **Interfaz Intuitiva**: Visualización clara del estado, confianza y progreso
- ✅ **Sistema Modular**: Fácil expansión de vocabulario
- ✅ **Multiplataforma**: Compatible con Windows y Linux Ubuntu 22.04

---

## Arquitectura del Sistema

### 1. Pipeline de Detección

```
Cámara Web → MediaPipe Hands → Extracción de Landmarks (63 features)
              ↓
         Secuencia Temporal (30 frames)
              ↓
         Modelo LSTM (clasificación)
              ↓
    Sistema de Estabilización (15 frames consecutivos)
              ↓
         Síntesis de Voz → Salida Auditiva
```

### 2. Modelo de Red Neuronal

**Arquitectura LSTM Profunda:**
```
Entrada: (30 frames, 63 features)
   ↓
LSTM Layer 1: 64 unidades + Dropout(0.2)
   ↓
BatchNormalization
   ↓
LSTM Layer 2: 128 unidades + Dropout(0.2)
   ↓
BatchNormalization
   ↓
LSTM Layer 3: 64 unidades + Dropout(0.2)
   ↓
Dense Layer: Softmax (7 clases)
```

**Características:**
- **Entrada**: Secuencias de 30 frames × 63 características
- **Características por frame**: 21 landmarks × 3 coordenadas (x, y, z)
- **Optimizador**: Adam
- **Función de pérdida**: Categorical Crossentropy
- **Regularización**: Dropout (20%) + BatchNormalization
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, TensorBoard

---

## Librerías y Tecnologías

### Procesamiento de Video y Visión por Computadora

**OpenCV 4.10.0.84**
- Captura de video desde cámara web
- Procesamiento de imágenes en tiempo real
- Conversión de espacios de color (BGR ↔ RGB)
- Renderización de interfaz gráfica y textos
- Detección de eventos de teclado

**MediaPipe 0.10.21**
- Detección de manos con modelo ML preentrenado
- Tracking de 21 landmarks anatómicos por mano
- Estimación de confianza de detección
- Procesamiento en CPU optimizado
- Normalización automática de coordenadas

### Aprendizaje Profundo

**TensorFlow 2.18.0**
- Framework principal de deep learning
- Compilación y entrenamiento del modelo
- Inferencia en tiempo real
- Optimización con oneDNN para CPU
- Soporte para operaciones tensoriales

**Keras 3.12**
- API de alto nivel para construcción del modelo
- Capas LSTM, Dense, Dropout, BatchNormalization
- Callbacks para entrenamiento adaptativo
- Serialización de modelos (.h5, .keras)

**NumPy 1.26.4**
- Operaciones matriciales y vectoriales
- Manipulación de arrays de landmarks
- Normalización y reshape de datos
- Gestión eficiente de secuencias temporales

**scikit-learn 1.6.0**
- Codificación de etiquetas (LabelEncoder)
- Conversión a representación categórica
- División de datasets (train/test split)
- Utilidades de preprocesamiento

### Síntesis de Voz

**pyttsx3 2.98** (Modo Offline)
- Text-to-Speech sin conexión a internet
- Soporte para voz en español
- Control de velocidad y volumen
- Ejecución síncrona y asíncrona
- Multiplataforma (Windows/Linux)

**gTTS 2.5.4** (Modo Online)
- Google Text-to-Speech API
- Voz natural de alta calidad
- Requiere conexión a internet
- Caché de archivos de audio
- Pronunciación en español optimizada

**pygame 2.6.1**
- Reproducción de archivos MP3 generados por gTTS
- Sistema de mixer para audio
- Control de reproducción (play/stop)

---

## Funcionamiento Detallado

### 1. Captura y Preprocesamiento

```python
# Captura de frame desde cámara
ret, frame = cap.read()

# Conversión de color BGR → RGB (MediaPipe requiere RGB)
image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Detección de manos con MediaPipe
results = hands.process(image_rgb)
```

### 2. Extracción de Características

Cada mano detectada proporciona **21 landmarks** (puntos anatómicos):

```
Landmarks de la mano:
0: Muñeca (WRIST)
1-4: Pulgar (THUMB_CMC, MCP, IP, TIP)
5-8: Índice (INDEX_FINGER_MCP, PIP, DIP, TIP)
9-12: Medio (MIDDLE_FINGER_MCP, PIP, DIP, TIP)
13-16: Anular (RING_FINGER_MCP, PIP, DIP, TIP)
17-20: Meñique (PINKY_MCP, PIP, DIP, TIP)
```

**Características por landmark:**
- `x`: Coordenada horizontal normalizada [0-1]
- `y`: Coordenada vertical normalizada [0-1]
- `z`: Profundidad relativa a la muñeca

**Total de características por frame: 21 × 3 = 63**

### 3. Detección Bilateral Inteligente

```python
# Si se detectan ambas manos
if len(results.multi_hand_landmarks) > 1:
    # Calcular confianza promedio de cada mano
    confidence_0 = np.mean([lm.x for lm in hand_0.landmark])
    confidence_1 = np.mean([lm.x for lm in hand_1.landmark])
    
    # Seleccionar la mano con mayor confianza
    selected_hand = hand_0 if confidence_0 > confidence_1 else hand_1
```

### 4. Creación de Secuencias Temporales

El modelo requiere **30 frames consecutivos** para analizar el movimiento:

```python
sequence_length = 30  # Ventana temporal
keypoints_sequence = []  # Buffer de 30 frames

# Acumular frames
keypoints_sequence.append(keypoints)  # 63 features

# Cuando tenemos 30 frames completos
if len(keypoints_sequence) == 30:
    # Preparar para predicción
    input_data = np.expand_dims(keypoints_sequence, axis=0)
    # Shape: (1, 30, 63)
```

### 5. Predicción con Modelo LSTM

```python
# Inferencia del modelo
prediction = model.predict(input_data, verbose=0)[0]

# Obtener clase con mayor probabilidad
predicted_class = np.argmax(prediction)
confidence = prediction[predicted_class]

# Mapear índice a nombre de seña
sign_name = class_names[predicted_class]
```

### 6. Sistema de Estabilización

Para evitar falsas detecciones, se implementa un mecanismo de estabilización:

```python
stability_threshold = 15  # Frames requeridos
stability_counter = 0

# Verificar confianza mínima
if confidence >= 0.66:  # 66% de confianza
    if sign_name == last_prediction:
        stability_counter += 1
    else:
        stability_counter = 1
        last_prediction = sign_name
    
    # Detección confirmada
    if stability_counter >= 15:
        confirmed_sign = sign_name
        # Activar síntesis de voz
        voice_system.speak_sync(confirmed_sign)
```

### 7. Sistema de Cooldown

Evita detecciones repetitivas inmediatas:

```python
cooldown_duration = 3.0  # segundos
last_detection_time = time.time()

# Verificar si ha pasado suficiente tiempo
if (current_time - last_detection_time) > cooldown_duration:
    # Permitir nueva detección
    can_detect = True
```

### 8. Visualización en Tiempo Real

```python
# Mostrar estado actual
if is_detecting:
    cv2.putText(frame, "PROCESANDO...", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

# Mostrar progreso de estabilización
progress_text = f"Estabilidad: {stability_counter}/15"
cv2.putText(frame, progress_text, (10, 200), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

# Mostrar confianza
confidence_text = f"Confianza: {confidence*100:.1f}%"
cv2.putText(frame, confidence_text, (10, 240), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Dibujar landmarks de la mano
mp_drawing.draw_landmarks(
    frame, 
    hand_landmarks,
    mp_hands.HAND_CONNECTIONS,
    mp_drawing_styles.get_default_hand_landmarks_style(),
    mp_drawing_styles.get_default_hand_connections_style()
)
```

---

## Módulos del Sistema

### 1. `detect_signs.py` - Detección en Tiempo Real

**Función principal**: Sistema completo de reconocimiento de señas

**Características:**
- Captura de video desde cámara
- Detección de manos con MediaPipe
- Predicción con modelo LSTM
- Síntesis de voz integrada
- Interfaz gráfica en tiempo real
- Sistema de estabilización y cooldown

**Controles:**
- `ENTER`: Activar/Desactivar detección
- `ESPACIO`: Forzar voz (si hay seña detectada)
- `Q`: Salir del programa

### 2. `collect_data.py` - Recolección de Datos

**Función principal**: Capturar datos de entrenamiento para nuevas señas

**Características:**
- Interfaz con diálogos Tkinter mejorados
- Captura de 30 secuencias por seña
- Almacenamiento automático en formato .npy
- Actualización de `signs.json`
- Opciones de continuar o salir

**Proceso:**
1. Ingresa nombre de la seña
2. Sistema prepara captura (cuenta regresiva)
3. Realiza 30 secuencias de 30 frames cada una
4. Guarda en carpeta `data/nombre_seña/`
5. Opción de capturar otra seña o salir

### 3. `train_model.py` - Entrenamiento del Modelo

**Función principal**: Entrenar modelo LSTM con datos recopilados

**Características:**
- Carga automática de datos desde carpeta `data/`
- Augmentación de datos con ruido gaussiano
- División train/test (80/20)
- Callbacks avanzados (EarlyStopping, ReduceLR)
- Logging con TensorBoard
- Guardado en formatos .h5 y .keras

**Proceso:**
1. Lee todas las carpetas en `data/`
2. Carga secuencias .npy
3. Aplica normalización y augmentación
4. Entrena modelo LSTM
5. Guarda modelo y actualiza signs.json

**Parámetros de entrenamiento:**
- Epochs: 200 (con early stopping)
- Batch size: 32
- Validation split: 20%
- Learning rate inicial: 0.001
- Paciencia (early stopping): 20 epochs

### 4. `voice_system.py` - Sistema de Síntesis de Voz

**Función principal**: Convertir texto a voz (TTS)

**Modos de operación:**

**Modo OFFLINE (pyttsx3):**
```python
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Velocidad
engine.setProperty('volume', 1.0)  # Volumen
engine.say(text)
engine.runAndWait()
```

**Modo ONLINE (gTTS):**
```python
tts = gTTS(text=text, lang='es', slow=False)
tts.save(temp_file)
pygame.mixer.music.load(temp_file)
pygame.mixer.music.play()
```

**Características:**
- Pre-generación de audio (caché)
- Control de repeticiones (cooldown)
- Modo síncrono y asíncrono
- Detección automática de voz en español
- Limpieza automática de archivos temporales

---

## Requisitos del Sistema

### Sistema Operativo
- **Windows 10/11** (recomendado)
- **Linux Ubuntu 22.04+**

### Hardware
- **Cámara web** (resolución mínima 720p @ 30fps)
- **CPU**: Intel Core i5 / AMD Ryzen 5 (4 núcleos) o superior
- **RAM**: 4 GB disponible (8 GB recomendado)
- **Espacio en disco**: 2 GB libre

### Software Base
- **Python 3.11.x** (IMPORTANTE: versión específica requerida)
- **Git** (opcional, para clonar el repositorio)

---

## Instalación

### Windows

```powershell
# Instalar Python 3.11 desde python.org
# Descargar: https://www.python.org/downloads/release/python-3119/

# Instalar librerías
pip install opencv-python==4.10.0.84
pip install mediapipe==0.10.21
pip install numpy==1.26.4
pip install tensorflow==2.18.0
pip install scikit-learn==1.6.0
pip install pyttsx3==2.98
pip install gtts==2.5.4
pip install pygame==2.6.1
```

### Ubuntu 22.04

**Opción 1: Script automático**
```bash
chmod +x install_ubuntu.sh
./install_ubuntu.sh
```

**Opción 2: Manual**
```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Python 3.11
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Dependencias del sistema
sudo apt install -y libopencv-dev python3-opencv libportaudio2 \
    portaudio19-dev espeak libespeak-dev ffmpeg libsm6 \
    libxext6 libxrender-dev libgomp1

# Crear entorno virtual
python3.11 -m venv venv
source venv/bin/activate

# Instalar librerías de Python
python3.11 -m pip install opencv-python==4.10.0.84 \
    mediapipe==0.10.21 numpy==1.26.4 tensorflow==2.18.0 \
    scikit-learn==1.6.0 pyttsx3==2.98 gtts==2.5.4 pygame==2.6.1
```

---

## Uso del Sistema

### 1. Ejecutar Detección de Señas

```bash
# Windows
py -3.11 detect_signs.py

# Linux
python3.11 detect_signs.py
```

**Proceso:**
1. Se abre ventana con feed de cámara
2. Presiona `ENTER` para activar detección
3. Realiza señas frente a la cámara
4. Sistema detecta y pronuncia la seña reconocida
5. Presiona `Q` para salir

### 2. Capturar Nuevas Señas

```bash
# Windows
py -3.11 collect_data.py

# Linux
python3.11 collect_data.py
```

**Proceso:**
1. Ingresa nombre de la seña nueva
2. Posiciónate frente a la cámara
3. Presiona `ENTER` para comenzar captura
4. Realiza la seña 30 veces
5. Datos guardados automáticamente
6. Opción de capturar otra seña

### 3. Entrenar Modelo con Nuevos Datos

```bash
# Windows
py -3.11 train_model.py

# Linux
python3.11 train_model.py
```

**Proceso:**
1. Sistema carga todos los datos de `data/`
2. Entrena modelo LSTM (puede tomar varios minutos)
3. Muestra métricas de accuracy y loss
4. Guarda modelo actualizado
5. Actualiza `signs.json` con nuevas señas

---

## Estructura de Archivos

```
SignLanguageDetection/
│
├── detect_signs.py           # Sistema principal de detección
├── collect_data.py            # Recolección de datos
├── train_model.py             # Entrenamiento del modelo
├── voice_system.py            # Sistema de síntesis de voz
│
├── sign_language_model.h5     # Modelo entrenado (formato HDF5)
├── sign_language_model.keras  # Modelo entrenado (formato Keras)
├── signs.json                 # Mapeo de índices a nombres de señas
│
├── data/                      # Datos de entrenamiento
│   ├── hola/                  # Seña "hola"
│   │   ├── seq_0.npy
│   │   ├── seq_1.npy
│   │   └── ... (30 archivos)
│   ├── adios/                 # Seña "adios"
│   ├── como_estas/
│   └── ...
│
├── logs/                      # Logs de TensorBoard
│   ├── train/
│   └── validation/
│
├── install_ubuntu.sh          # Script de instalación Ubuntu
├── comandos_ubuntu.txt        # Comandos de instalación manual
└── README.md                  # Este archivo
```

---

## Configuración Avanzada

### Ajustar Precisión de Detección

En `detect_signs.py`:

```python
# Línea ~25
CONFIDENCE_THRESHOLD = 0.66  # Cambiar entre 0.5 y 0.9

# Valores sugeridos:
# 0.5 - 0.6: Mayor sensibilidad, más detecciones, posibles falsos positivos
# 0.66: Balanceado (recomendado)
# 0.7 - 0.9: Mayor precisión, menos detecciones, más confiable
```

### Ajustar Estabilización

```python
# Línea ~30
STABILITY_THRESHOLD = 15  # Frames consecutivos requeridos

# Valores sugeridos:
# 10: Detección más rápida pero menos estable
# 15: Balanceado (recomendado)
# 20-25: Mayor estabilidad pero más lento
```

### Ajustar Cooldown

```python
# Línea ~35
COOLDOWN_DURATION = 3.0  # Segundos entre detecciones

# Valores sugeridos:
# 1.5 - 2.0: Para practicar señas rápidas
# 3.0: Balanceado (recomendado)
# 4.0 - 5.0: Para demostraciones o presentaciones
```

---

## Solución de Problemas

### Error: No se detecta la cámara

```python
# Cambiar índice de cámara en detect_signs.py
cap = cv2.VideoCapture(0)  # Probar con 1, 2, etc.
```

### Error: Modelo no encontrado

```bash
# Verificar que existe el archivo
ls sign_language_model.h5

# Si no existe, entrenar el modelo
python3.11 train_model.py
```

### Error: Voz no funciona (Linux)

```bash
# Instalar espeak
sudo apt install espeak libespeak-dev

# Reinstalar pyttsx3
pip install --upgrade pyttsx3
```

### Error: ImportError con TensorFlow

```bash
# Reinstalar TensorFlow
pip uninstall tensorflow
pip install tensorflow==2.18.0
```

### Error: Detecciones inconsistentes

1. **Mejorar iluminación**: Usar luz frontal uniforme
2. **Limpiar lente**: Cámara sin obstrucciones
3. **Fondo neutro**: Evitar objetos en movimiento
4. **Distancia adecuada**: 50-80 cm de la cámara
5. **Recapturar datos**: Más variedad en los datos de entrenamiento

---

## Rendimiento y Optimización

### Requisitos Mínimos

- **CPU**: Intel Core i5 / AMD Ryzen 5 (4 núcleos)
- **RAM**: 4 GB disponible
- **Cámara**: Webcam 720p @ 30fps
- **Sistema Operativo**: Windows 10/11 o Ubuntu 20.04+
- **Conexión a Internet**: Solo para modo online (gTTS)

### Rendimiento Esperado

- **Latencia de detección**: 100-150 ms
- **FPS de procesamiento**: 25-30 fps
- **Tiempo de respuesta de voz**: 200-300 ms (offline), 500-800 ms (online)
- **Uso de CPU**: 40-60% (un núcleo)
- **Uso de RAM**: 500-800 MB

### Optimizaciones Aplicadas

1. **Inferencia optimizada**: `model.predict()` con verbose=0
2. **oneDNN habilitado**: Operaciones CPU aceleradas
3. **Pre-generación de audio**: Caché para señas comunes
4. **Detección selectiva**: Cooldown para reducir carga
5. **Buffer circular**: Gestión eficiente de secuencias

---

## Documentación Técnica Detallada

### Sistema Completo: Flujo de Trabajo

El sistema de reconocimiento de lenguaje de señas está compuesto por tres fases principales que trabajan de manera integrada:

#### **FASE 1: Recolección de Datos** (`collect_data.py`)

Esta fase es fundamental para entrenar el modelo con nuevas señas. El proceso es completamente automatizado y guiado.

**1.1 Interfaz de Usuario Mejorada**

El sistema utiliza Tkinter para crear diálogos profesionales:

```python
def capture_new_sign():
    # Crear ventana principal con estilo
    root = tk.Tk()
    root.withdraw()  # Ocultar ventana principal
    
    # Diálogo personalizado con título y mensaje
    dialog = tk.Toplevel(root)
    dialog.title("Captura de Nueva Seña")
    
    # Mensaje con instrucciones claras
    message = """
    ╔══════════════════════════════════════╗
    ║  CAPTURA DE NUEVA SEÑA               ║
    ╚══════════════════════════════════════╝
    
    Ingresa el nombre de la seña que deseas capturar.
    Ejemplo: 'hola', 'gracias', 'como_estas'
    """
```

**1.2 Proceso de Captura Frame por Frame**

El sistema captura **30 secuencias** de **30 frames** cada una, totalizando **900 frames** por seña:

```python
SEQUENCE_LENGTH = 30  # Frames por secuencia
NUM_SEQUENCES = 30    # Total de secuencias

for sequence in range(NUM_SEQUENCES):
    keypoints_sequence = []
    
    # Capturar 30 frames consecutivos
    for frame_num in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        
        # Convertir BGR → RGB para MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar manos
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Extraer landmarks de la primera mano
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Convertir a array numpy (21 puntos × 3 coords = 63 features)
            keypoints = np.array([[lm.x, lm.y, lm.z] 
                                  for lm in hand_landmarks.landmark]).flatten()
        else:
            # Si no hay mano detectada, usar ceros
            keypoints = np.zeros(63)
        
        keypoints_sequence.append(keypoints)
    
    # Guardar secuencia completa en formato .npy
    np.save(f'data/{sign_name}/seq_{sequence}.npy', keypoints_sequence)
```

**1.3 Visualización en Tiempo Real**

Durante la captura, el usuario ve:
- **Contador de progreso**: "Capturando: 15/30"
- **Frame actual**: "Frame: 10/30"
- **Landmarks dibujados**: Visualización de los 21 puntos de la mano
- **Instrucciones**: Mensajes claros de qué hacer

```python
# Mostrar información en pantalla
cv2.putText(frame, f"Secuencia: {sequence+1}/{NUM_SEQUENCES}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(frame, f"Frame: {frame_num+1}/{SEQUENCE_LENGTH}", 
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# Dibujar landmarks de la mano
mp_drawing.draw_landmarks(
    frame, 
    hand_landmarks,
    mp_hands.HAND_CONNECTIONS,
    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
)
```

**1.4 Almacenamiento Estructurado**

Los datos se organizan automáticamente:

```
data/
├── hola/
│   ├── seq_0.npy   # Shape: (30, 63)
│   ├── seq_1.npy   # 30 frames × 63 features
│   ├── seq_2.npy
│   └── ...
│   └── seq_29.npy
├── adios/
│   ├── seq_0.npy
│   └── ...
└── como_estas/
    ├── seq_0.npy
    └── ...
```

Cada archivo `.npy` contiene:
- **Dimensiones**: (30 frames, 63 features)
- **Contenido**: Coordenadas x, y, z de 21 landmarks por frame
- **Formato**: NumPy array para carga rápida

**1.5 Actualización de Mapeo**

Al finalizar la captura, se actualiza `signs.json`:

```python
# Leer mapeo actual
with open('signs.json', 'r') as f:
    signs_data = json.load(f)

# Agregar nueva seña con índice único
new_index = max(signs_data.keys()) + 1
signs_data[new_index] = sign_name

# Guardar actualización
with open('signs.json', 'w') as f:
    json.dump(signs_data, f, indent=4)
```

Ejemplo de `signs.json`:
```json
{
    "0": "hola",
    "1": "adios",
    "2": "gracias",
    "3": "como_estas",
    "4": "bien_hecho"
}
```

---

#### **FASE 2: Entrenamiento del Modelo** (`train_model.py`)

Esta fase transforma los datos capturados en un modelo LSTM capaz de reconocer señas.

**2.1 Carga y Preparación de Datos**

```python
def load_data():
    sequences = []
    labels = []
    
    # Obtener todas las carpetas en data/
    sign_folders = os.listdir('data/')
    
    for label_idx, sign_name in enumerate(sign_folders):
        sign_path = f'data/{sign_name}'
        
        # Cargar todas las secuencias de esta seña
        for seq_file in os.listdir(sign_path):
            # Cargar archivo .npy
            sequence = np.load(f'{sign_path}/{seq_file}')
            
            sequences.append(sequence)
            labels.append(label_idx)
    
    # Convertir a arrays numpy
    X = np.array(sequences)  # Shape: (num_samples, 30, 63)
    y = np.array(labels)      # Shape: (num_samples,)
    
    return X, y, sign_folders
```

**2.2 Augmentación de Datos**

Para mejorar la generalización, se aplica ruido gaussiano:

```python
def augment_data(X, y, augmentation_factor=2):
    """
    Genera versiones adicionales de los datos con ruido
    """
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X)):
        # Datos originales
        X_augmented.append(X[i])
        y_augmented.append(y[i])
        
        # Generar copias con ruido
        for _ in range(augmentation_factor):
            # Ruido gaussiano pequeño (desviación 0.01)
            noise = np.random.normal(0, 0.01, X[i].shape)
            augmented_sequence = X[i] + noise
            
            # Clip para mantener valores en rango válido [0, 1]
            augmented_sequence = np.clip(augmented_sequence, 0, 1)
            
            X_augmented.append(augmented_sequence)
            y_augmented.append(y[i])
    
    return np.array(X_augmented), np.array(y_augmented)
```

**2.3 Arquitectura del Modelo LSTM**

El modelo es una red neuronal recurrente profunda:

```python
def build_model(input_shape, num_classes):
    """
    Construye modelo LSTM con 3 capas
    
    Args:
        input_shape: (30, 63) - 30 frames × 63 features
        num_classes: Número de señas a clasificar
    """
    model = Sequential([
        # Primera capa LSTM: 64 unidades
        LSTM(64, return_sequences=True, activation='relu', 
             input_shape=input_shape),
        Dropout(0.2),  # 20% dropout para regularización
        BatchNormalization(),  # Normalización de batch
        
        # Segunda capa LSTM: 128 unidades (capa más profunda)
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.2),
        BatchNormalization(),
        
        # Tercera capa LSTM: 64 unidades
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.2),
        
        # Capa densa de salida con softmax
        Dense(num_classes, activation='softmax')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Explicación de la Arquitectura:**

1. **LSTM Layer 1 (64 unidades)**:
   - Procesa secuencias temporales de 30 frames
   - `return_sequences=True`: Pasa secuencia completa a siguiente capa
   - `activation='relu'`: Función de activación para no-linealidad

2. **Dropout (0.2)**:
   - Desactiva aleatoriamente 20% de neuronas durante entrenamiento
   - Previene overfitting (sobreajuste)

3. **BatchNormalization**:
   - Normaliza activaciones entre batches
   - Acelera entrenamiento y mejora convergencia

4. **LSTM Layer 2 (128 unidades)**:
   - Capa más profunda para capturar patrones complejos
   - Aprende representaciones de alto nivel

5. **LSTM Layer 3 (64 unidades)**:
   - `return_sequences=False`: Solo devuelve último estado
   - Condensa información temporal en vector fijo

6. **Dense Layer (num_classes unidades)**:
   - Capa de clasificación final
   - `softmax`: Convierte salidas en probabilidades

**2.4 Callbacks de Entrenamiento**

El sistema utiliza callbacks avanzados para optimizar el entrenamiento:

```python
# Early Stopping: Detiene si no mejora
early_stopping = EarlyStopping(
    monitor='val_loss',        # Monitorear pérdida de validación
    patience=20,               # Esperar 20 epochs sin mejora
    restore_best_weights=True, # Restaurar mejores pesos
    verbose=1
)

# Reduce Learning Rate: Ajusta tasa de aprendizaje
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # Monitorear pérdida de validación
    factor=0.5,            # Reducir LR a la mitad
    patience=10,           # Esperar 10 epochs
    min_lr=1e-7,          # LR mínimo
    verbose=1
)

# TensorBoard: Logging de métricas
tensorboard = TensorBoard(
    log_dir='logs/',
    histogram_freq=1,      # Guardar histogramas cada epoch
    write_graph=True,      # Guardar gráfico del modelo
    update_freq='epoch'    # Actualizar por epoch
)
```

**2.5 Entrenamiento**

```python
# Preparar datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convertir etiquetas a categórico
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Entrenar modelo
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=200,              # Máximo 200 epochs
    batch_size=32,           # 32 muestras por batch
    callbacks=[early_stopping, reduce_lr, tensorboard],
    verbose=1
)
```

**Proceso durante el Entrenamiento:**

```
Epoch 1/200
█████████████████████ 890/890 [00:15<00:00] - loss: 1.4521 - accuracy: 0.4234 - val_loss: 1.2156 - val_accuracy: 0.5123
Epoch 2/200
█████████████████████ 890/890 [00:14<00:00] - loss: 0.9876 - accuracy: 0.6543 - val_loss: 0.8234 - val_accuracy: 0.7012
...
Epoch 45/200
█████████████████████ 890/890 [00:14<00:00] - loss: 0.1234 - accuracy: 0.9612 - val_loss: 0.1567 - val_accuracy: 0.9445

Early stopping triggered. Best weights restored from epoch 25.
```

**2.6 Guardado del Modelo**

El modelo se guarda en dos formatos:

```python
# Formato HDF5 (compatible con versiones antiguas)
model.save('sign_language_model.h5')

# Formato Keras nativo (recomendado)
model.save('sign_language_model.keras')

print("[GUARDADO] Modelo guardado exitosamente")
print(f"  - Archivo HDF5: sign_language_model.h5")
print(f"  - Archivo Keras: sign_language_model.keras")
print(f"  - Número de clases: {num_classes}")
print(f"  - Accuracy final: {accuracy:.2%}")
```

---

#### **FASE 3: Detección en Tiempo Real** (`detect_signs.py`)

Esta es la fase de producción donde el sistema reconoce señas en vivo.

**3.1 Inicialización del Sistema**

```python
# Cargar modelo entrenado
model = load_model('sign_language_model.h5', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Cargar mapeo de clases
with open('signs.json', 'r') as f:
    signs_data = json.load(f)
    class_names = list(signs_data.values())

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,      # Modo video
    max_num_hands=2,               # Detectar hasta 2 manos
    min_detection_confidence=0.5,  # 50% confianza mínima
    min_tracking_confidence=0.5
)

# Inicializar sistema de voz
voice_system = VoiceSystem()

# Inicializar cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
```

**3.2 Loop Principal de Detección**

```python
# Variables de estado
keypoints_sequence = []           # Buffer de 30 frames
is_detecting = False              # Estado de detección
stability_counter = 0             # Contador de estabilidad
last_prediction = None            # Última predicción
confirmed_sign = None             # Seña confirmada
in_cooldown = False               # Estado de cooldown
cooldown_start_time = 0           # Inicio del cooldown
prediction_history = deque(maxlen=8)  # Historial de predicciones

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Voltear frame horizontalmente (efecto espejo)
    frame = cv2.flip(frame, 1)
    
    # Convertir a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar con MediaPipe
    results = hands.process(image_rgb)
    
    # ... (procesamiento continúa)
```

**3.3 Procesamiento de Manos Detectadas**

```python
if results.multi_hand_landmarks and is_detecting:
    # Si hay múltiples manos, seleccionar la mejor
    if len(results.multi_hand_landmarks) > 1:
        # Calcular confianza de cada mano
        hand_confidences = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Promedio de visibilidad de landmarks
            confidence = np.mean([lm.visibility if hasattr(lm, 'visibility') 
                                  else 1.0 for lm in hand_landmarks.landmark])
            hand_confidences.append(confidence)
        
        # Seleccionar mano con mayor confianza
        best_hand_idx = np.argmax(hand_confidences)
        selected_hand = results.multi_hand_landmarks[best_hand_idx]
    else:
        selected_hand = results.multi_hand_landmarks[0]
    
    # Extraer keypoints (63 features)
    keypoints = np.array([[lm.x, lm.y, lm.z] 
                          for lm in selected_hand.landmark]).flatten()
    
    # Agregar a secuencia
    keypoints_sequence.append(keypoints)
    
    # Mantener solo últimos 30 frames
    if len(keypoints_sequence) > 30:
        keypoints_sequence = keypoints_sequence[-30:]
```

**3.4 Predicción y Estabilización**

```python
# Cuando tenemos 30 frames completos
if len(keypoints_sequence) == 30 and not in_cooldown:
    # Preparar entrada para el modelo
    input_data = np.expand_dims(keypoints_sequence, axis=0)
    # Shape: (1, 30, 63)
    
    # Hacer predicción
    prediction = model.predict(input_data, verbose=0)[0]
    
    # Obtener clase con mayor probabilidad
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    sign_name = class_names[predicted_class]
    
    # Agregar a historial
    prediction_history.append((sign_name, confidence))
    
    # Verificar confianza mínima
    if confidence >= CONFIDENCE_THRESHOLD:  # 0.66
        # Verificar consistencia en historial
        recent_predictions = [p[0] for p in list(prediction_history)[-5:]]
        most_common = max(set(recent_predictions), key=recent_predictions.count)
        
        if sign_name == most_common:
            # Incrementar contador de estabilidad
            if sign_name == last_prediction:
                stability_counter += 1
            else:
                stability_counter = 1
                last_prediction = sign_name
            
            # Mostrar progreso
            print(f"[DETECCION] {sign_name} - Confianza: {confidence:.2f}, "
                  f"Estable: {stability_counter}/{STABILITY_THRESHOLD}")
            
            # Si alcanza estabilidad requerida
            if stability_counter >= STABILITY_THRESHOLD:  # 15 frames
                confirmed_sign = sign_name
                
                # Activar síntesis de voz
                voice_system.speak_sync(confirmed_sign)
                
                print(f"[DETECTADO] {confirmed_sign} "
                      f"(confianza: {confidence:.2f}, estabilidad: {stability_counter})")
                
                # Iniciar cooldown
                in_cooldown = True
                cooldown_start_time = time.time()
                
                # Resetear contadores
                stability_counter = 0
                last_prediction = None
                keypoints_sequence = []
        else:
            # Predicción inconsistente, resetear
            stability_counter = 0
    else:
        # Confianza baja, resetear
        stability_counter = 0
```

**3.5 Sistema de Cooldown**

```python
# Verificar si el cooldown ha terminado
if in_cooldown:
    elapsed_time = time.time() - cooldown_start_time
    
    if elapsed_time >= COOLDOWN_DURATION:  # 3.0 segundos
        in_cooldown = False
        print("[SISTEMA] Cooldown terminado - Listo para nueva deteccion")
    else:
        # Mostrar tiempo restante
        remaining = COOLDOWN_DURATION - elapsed_time
        cv2.putText(frame, f"Cooldown: {remaining:.1f}s", 
                    (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (0, 165, 255), 2)
```

**3.6 Visualización Completa**

```python
# Título principal
cv2.putText(frame, "RECONOCIMIENTO DE SENAS", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

# Estado de detección
if is_detecting:
    status_text = "PROCESANDO..."
    status_color = (0, 255, 255)  # Amarillo
else:
    status_text = "PAUSADO"
    status_color = (0, 0, 255)  # Rojo

cv2.putText(frame, status_text, (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

# Mostrar predicción actual
if last_prediction and is_detecting:
    cv2.putText(frame, f"ANALIZANDO: {last_prediction}", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2)
    
    # Progreso de estabilidad
    progress_text = f"Estabilidad: {stability_counter}/{STABILITY_THRESHOLD}"
    cv2.putText(frame, progress_text, (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Confianza
    confidence_text = f"Confianza: {confidence*100:.1f}%"
    conf_color = (0, 255, 0) if confidence >= 0.75 else (255, 165, 0)
    cv2.putText(frame, confidence_text, (10, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)

# Mostrar seña confirmada
if confirmed_sign and in_cooldown:
    cv2.putText(frame, f"DETECTADO: {confirmed_sign}", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

# Dibujar landmarks
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

# Instrucciones
cv2.putText(frame, "ENTER: Activar/Pausar | Q: Salir | ESPACIO: Forzar voz",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

# Mostrar frame
cv2.imshow('Reconocimiento de Lenguaje de Senas', frame)
```

**3.7 Control de Teclado**

```python
key = cv2.waitKey(1) & 0xFF

if key == ord('q') or key == ord('Q'):
    # Salir del programa
    print("[SISTEMA] Cerrado correctamente")
    break

elif key == 13:  # ENTER
    # Activar/Desactivar detección
    is_detecting = not is_detecting
    status = "ACTIVADA" if is_detecting else "PAUSADA"
    print(f"[SISTEMA] DETECCION {status}")
    
    # Limpiar buffers al cambiar estado
    keypoints_sequence = []
    stability_counter = 0
    last_prediction = None
    prediction_history.clear()

elif key == 32:  # ESPACIO
    # Forzar síntesis de voz si hay seña detectada
    if last_prediction:
        voice_system.speak_sync(last_prediction)
        print(f"[VOZ] Forzada reproducción de: {last_prediction}")
```

---

### Sistema de Síntesis de Voz Detallado

El módulo `voice_system.py` maneja la conversión de texto a voz con dos modos.

**Modo OFFLINE (pyttsx3)**

```python
class VoiceSystem:
    def __init__(self):
        self.engine = pyttsx3.init()
        
        # Configurar velocidad (palabras por minuto)
        self.engine.setProperty('rate', 150)
        
        # Configurar volumen (0.0 a 1.0)
        self.engine.setProperty('volume', 1.0)
        
        # Buscar voz en español
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'spanish' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                print(f"[VOZ] Voz española encontrada: {voice.name}")
                break
    
    def speak_sync(self, text):
        """Habla de forma síncrona (bloquea hasta terminar)"""
        if self.is_speaking:
            return False
        
        try:
            self.is_speaking = True
            
            # Limpiar texto (reemplazar _ por espacio)
            clean_text = text.replace("_", " ")
            
            # Sintetizar voz
            self.engine.say(clean_text)
            self.engine.runAndWait()
            
            return True
        except Exception as e:
            print(f"[ERROR VOZ] {e}")
            return False
        finally:
            self.is_speaking = False
```

**Modo ONLINE (gTTS + pygame)**

```python
class VoiceSystem:
    def __init__(self):
        pygame.mixer.init()
        self.audio_cache = {}
        
        # Pre-generar archivos de audio
        self.preload_words()
    
    def preload_words(self):
        """Pre-genera MP3 para todas las señas"""
        with open('signs.json', 'r') as f:
            signs_data = json.load(f)
            words = list(signs_data.values())
        
        for word in words:
            clean_word = word.replace("_", " ")
            
            # Generar audio con Google TTS
            tts = gTTS(text=clean_word, lang='es', slow=False)
            
            # Guardar en archivo temporal
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file.close()
            tts.save(temp_file.name)
            
            # Guardar ruta en caché
            self.audio_cache[word] = temp_file.name
        
        print(f"[VOZ] {len(self.audio_cache)} archivos de audio pre-generados")
    
    def speak_sync(self, text):
        """Reproduce audio desde caché"""
        if text in self.audio_cache:
            audio_file = self.audio_cache[text]
        else:
            # Generar on-the-fly si no está en caché
            clean_text = text.replace("_", " ")
            tts = gTTS(text=clean_text, lang='es', slow=False)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file.close()
            tts.save(temp_file.name)
            
            audio_file = temp_file.name
            self.audio_cache[text] = audio_file
        
        # Reproducir con pygame
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        # Esperar a que termine
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
```

---

### Optimizaciones y Mejores Prácticas

**1. Gestión Eficiente de Memoria**

```python
# Usar deque con tamaño máximo (elimina automáticamente elementos antiguos)
from collections import deque
keypoints_sequence = deque(maxlen=30)
prediction_history = deque(maxlen=8)

# Liberar recursos al salir
cap.release()
cv2.destroyAllWindows()
voice_system.cleanup()
hands.close()
```

**2. Procesamiento Optimizado**

```python
# Reducir verbosidad de predicción
prediction = model.predict(input_data, verbose=0)

# Procesar solo cuando es necesario
if is_detecting and results.multi_hand_landmarks:
    # Procesar...
else:
    # Saltear procesamiento
    pass
```

**3. Manejo Robusto de Errores**

```python
try:
    # Cargar modelo
    model = load_model('sign_language_model.h5')
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")
    print("[INFO] Ejecuta 'train_model.py' primero")
    exit(1)

try:
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("No se pudo acceder a la cámara")
except Exception as e:
    print(f"[ERROR] Problema con la cámara: {e}")
    exit(1)
```

**4. Logging Informativo**

```python
# Usar prefijos claros
print("[SISTEMA] Iniciando...")
print("[VOZ] Audio pre-generado")
print("[MODELO] Cargado exitosamente")
print("[DETECCION] hola - Confianza: 0.85")
print("[ERROR] Problema en la cámara")
print("[ADVERTENCIA] Confianza baja")
```

---

### Parámetros Configurables

Todos los parámetros importantes están centralizados para fácil ajuste:

```python
# === CONFIGURACIÓN DE DETECCIÓN ===
CONFIDENCE_THRESHOLD = 0.66      # Confianza mínima (66%)
STABILITY_THRESHOLD = 15         # Frames consecutivos requeridos
COOLDOWN_DURATION = 3.0          # Segundos entre detecciones
HISTORY_SIZE = 8                 # Tamaño del historial de predicciones

# === CONFIGURACIÓN DE MEDIAPIPE ===
MAX_NUM_HANDS = 2                # Máximo de manos a detectar
MIN_DETECTION_CONFIDENCE = 0.5   # Confianza mínima de detección
MIN_TRACKING_CONFIDENCE = 0.5    # Confianza mínima de tracking

# === CONFIGURACIÓN DE SECUENCIAS ===
SEQUENCE_LENGTH = 30             # Frames por secuencia
NUM_SEQUENCES = 30               # Secuencias por seña

# === CONFIGURACIÓN DE ENTRENAMIENTO ===
EPOCHS = 200                     # Máximo de epochs
BATCH_SIZE = 32                  # Tamaño de batch
VALIDATION_SPLIT = 0.2           # 20% para validación
LEARNING_RATE = 0.001            # Tasa de aprendizaje inicial
EARLY_STOPPING_PATIENCE = 20     # Epochs sin mejora antes de parar

# === CONFIGURACIÓN DE VOZ ===
VOICE_RATE = 150                 # Palabras por minuto
VOICE_VOLUME = 1.0               # Volumen (0.0 a 1.0)
VOICE_LANG = 'es'                # Idioma (español)
```

---

### Flujo Completo del Sistema

**Diagrama de Estados:**

```
[INICIO]
   ↓
[Cargar Modelo] → [Error] → [SALIR]
   ↓
[Inicializar Cámara] → [Error] → [SALIR]
   ↓
[Inicializar MediaPipe]
   ↓
[Inicializar Voz]
   ↓
[Estado: PAUSADO]
   ↓
[Usuario presiona ENTER]
   ↓
[Estado: DETECTANDO]
   ↓
[Capturar Frame] → [Convertir RGB] → [Detectar Manos]
   ↓                                         ↓
[Manos no detectadas]             [Manos detectadas]
   ↓                                         ↓
[Volver a capturar]              [Extraer landmarks]
                                             ↓
                                   [Agregar a secuencia]
                                             ↓
                                   [¿30 frames completos?]
                                             ↓
                                          [SÍ]
                                             ↓
                                   [Hacer predicción]
                                             ↓
                                   [¿Confianza >= 66%?]
                                             ↓
                                          [SÍ]
                                             ↓
                                   [¿Consistente en historial?]
                                             ↓
                                          [SÍ]
                                             ↓
                                   [Incrementar estabilidad]
                                             ↓
                                   [¿Estabilidad >= 15?]
                                             ↓
                                          [SÍ]
                                             ↓
                                   [SEÑA CONFIRMADA]
                                             ↓
                                   [Sintetizar voz]
                                             ↓
                                   [Iniciar cooldown (3s)]
                                             ↓
                                   [Esperar cooldown]
                                             ↓
                                   [Volver a detectar]
```

---

Esta documentación técnica detallada cubre todos los aspectos del sistema, desde la recolección de datos hasta la detección en tiempo real, incluyendo el funcionamiento interno de cada módulo y las decisiones de diseño tomadas.



### Proceso Completo

1. **Capturar nueva seña:**
   ```bash
   python3.11 collect_data.py
   ```

2. **Verificar datos guardados:**
   ```bash
   ls data/nueva_sena/
   # Debe mostrar seq_0.npy hasta seq_29.npy
   ```

3. **Entrenar modelo actualizado:**
   ```bash
   python3.11 train_model.py
   ```

4. **Probar nueva seña:**
   ```bash
   python3.11 detect_signs.py
   ```

### Recomendaciones para Captura

- **Variabilidad**: Realizar la seña con ligeras variaciones
- **Velocidad**: Combinar ejecuciones rápidas y lentas
- **Ángulos**: Capturar desde diferentes perspectivas
- **Iluminación**: Grabar en distintas condiciones de luz
- **Personas**: Idealmente, múltiples personas ejecutando la seña

---

## Métricas y Evaluación

### Métricas del Modelo

El sistema proporciona:
- **Accuracy**: Precisión general de clasificación
- **Loss**: Función de pérdida (categorical crossentropy)
- **Confianza por predicción**: Probabilidad softmax
- **Estabilidad**: Consistencia en frames consecutivos

### Visualización con TensorBoard

```bash
# Iniciar TensorBoard
tensorboard --logdir=logs

# Abrir navegador en:
# http://localhost:6006
```

**Gráficas disponibles:**
- Accuracy vs Epochs (train/validation)
- Loss vs Epochs (train/validation)
- Learning rate schedule
- Histogramas de pesos

---

## Tecnologías y Frameworks

| Categoría | Tecnología | Versión | Propósito |
|-----------|------------|---------|-----------|
| Lenguaje | Python | 3.11.9 | Base del sistema |
| Deep Learning | TensorFlow | 2.18.0 | Framework principal |
| API Alto Nivel | Keras | 3.12 | Construcción del modelo |
| Visión por Computadora | OpenCV | 4.10.0.84 | Procesamiento de video |
| Detección de Manos | MediaPipe | 0.10.21 | Landmarks de manos |
| Computación Numérica | NumPy | 1.26.4 | Operaciones matriciales |
| Machine Learning | scikit-learn | 1.6.0 | Preprocesamiento |
| TTS Offline | pyttsx3 | 2.98 | Síntesis de voz sin internet |
| TTS Online | gTTS | 2.5.4 | Google Text-to-Speech |
| Audio | pygame | 2.6.1 | Reproducción de audio |

---

## Contribuciones y Desarrollo

### Áreas de Mejora Futuras

1. **Expansión de vocabulario**: Más señas y frases complejas
2. **Detección de frases**: Secuencias de múltiples señas
3. **Reconocimiento de contexto**: Gramática de lenguaje de señas
4. **Interfaz web**: Versión accesible desde navegador
5. **Modelo más ligero**: Optimización para dispositivos móviles
6. **Multi-idioma**: Soporte para diferentes lenguajes de señas (ASL, LSE, etc.)

---

## Licencia

Este proyecto es de código abierto y está disponible para uso educativo y de investigación.

---

## Contacto y Soporte

Para reportar problemas, sugerencias o contribuciones:
- **GitHub**: [Josu-F1/SignLanguageDetection](https://github.com/Josu-F1/SignLanguageDetection)
- **Branch principal**: `ramaProbar`

---

## Referencias

1. MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands
2. TensorFlow LSTM: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
3. OpenCV Documentation: https://docs.opencv.org/
4. Sign Language Recognition Papers: IEEE Xplore

---

**Última actualización**: 21 de noviembre de 2025  
**Versión del sistema**: 1.0  
**Compatibilidad**: Windows 10/11, Ubuntu 22.04+

# O descargar desde python.org
```

#### En Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-dev
```

### Paso 2: Clonar o Descargar el Proyecto

#### Opción A: Con Git
```bash
git clone https://github.com/Josu-F1/SignLanguageDetection.git
cd SignLanguageDetection
```

#### Opción B: Descarga directa
1. Descargar ZIP del proyecto
2. Extraer en una carpeta
3. Abrir terminal en esa carpeta

### Paso 3: Instalar Dependencias

#### Método Automático (Recomendado):
```bash
# Windows
python -m pip install -r requirements.txt

# macOS/Linux
python3.11 -m pip install -r requirements.txt
```

#### Método Manual (si falla el automático):
```bash
# Librerías principales
pip install opencv-python==4.10.0.84
pip install mediapipe==0.10.21
pip install tensorflow==2.17.0
pip install keras==3.12.0

# Procesamiento de datos
pip install numpy==1.26.4
pip install scikit-learn==1.7.2
pip install scipy==1.16.3

# Síntesis de voz
pip install pyttsx3==2.99
pip install gTTS==2.5.4
pip install pygame==2.6.1

# Visualización y análisis
pip install matplotlib==3.10.7
pip install seaborn==0.13.2
pip install pandas==2.3.3

# Interfaz gráfica
pip install tkinter  # Ya viene con Python

# Utilidades
pip install requests==2.32.5
```

### Paso 4: Verificar Instalación

```bash
python -c "import cv2, mediapipe, tensorflow, pyttsx3, pygame; print('✅ Todas las librerías instaladas correctamente')"
```

## 📁 Estructura del Proyecto

```
SignLanguageDetection/
├── 📄 README.md                    # Este archivo
├── 📄 requirements.txt             # Dependencias del proyecto
├── 📄 backup_project.py            # Script de backup/restauración
│
├── 🎯 Scripts Principales:
│   ├── 📄 detect_signs.py          # 🔥 Script principal - Detección en tiempo real
│   ├── 📄 collect_data.py          # 📊 Recolección de datos para entrenar
│   ├── 📄 train_model.py           # 🧠 Entrenamiento del modelo
│   ├── 📄 manage_signs.py          # 🗂️ Gestor de señas (agregar/eliminar/renombrar)
│   └── 📄 voice_system.py          # 🔊 Sistema de síntesis de voz
│
├── 🗂️ Datos y Modelos:
│   ├── 📁 data/                    # Datos de entrenamiento por seña
│   │   ├── 📁 hola/               # Secuencias para "hola"
│   │   ├── 📁 adios/              # Secuencias para "adios"
│   │   └── 📁 [otras_señas]/      # Más señas...
│   │
│   ├── 📄 sign_language_model.keras   # Modelo entrenado
│   ├── 📄 signs.json              # Mapeo de índices a nombres de señas
│   └── 📄 training_stats.json     # Estadísticas del último entrenamiento
│
└── 📁 logs/                        # Logs de TensorBoard (generados automáticamente)
```

## 🎮 Uso del Sistema

### 1. 🎬 Recolectar Datos (Primera vez o nuevas señas)

```bash
python collect_data.py
```

**Instrucciones:**
- Ingresa el nombre de la seña cuando se solicite
- Haz la seña de forma clara y consistente
- Mantén las manos visibles en todo momento
- El sistema grabará 40 secuencias de 30 frames cada una
- Presiona `Q` para continuar, `ESC` para cancelar

### 2. 🧠 Entrenar el Modelo

```bash
python train_model.py
```

**El sistema:**
- Analiza automáticamente la calidad de los datos
- Filtra secuencias inválidas
- Entrena un modelo LSTM profundo
- Genera reportes de precisión
- Guarda el modelo y actualiza `signs.json`

### 3. 🎯 Detectar Señas en Tiempo Real

```bash
python detect_signs.py
```

**Controles:**
- `Q` - Salir del programa
- `ESPACIO` - Forzar reproducción de voz
- Mantén las señas 2-3 segundos para mejor detección

### 4. 🛡️ Hacer Backup del Proyecto

```bash
python backup_project.py
```

**Para restaurar:**
```bash
python backup_project.py restore backup_YYYYMMDD_HHMMSS
```

### 5. 🗂️ Gestionar Señas (Agregar/Eliminar/Renombrar)

```bash
python manage_signs.py
```

El **Gestor de Señas** te permite administrar fácilmente las palabras del sistema:

#### 📋 **Funciones Disponibles:**

- **📋 Listar señas** - Ver todas las señas con su estado
- **🗑️ Eliminar seña** - Borra datos y actualiza JSON automáticamente
- **✏️ Renombrar seña** - Cambia nombres manteniendo sincronización
- **➕ Agregar nueva seña** - Crea entradas para recopilar datos después
- **🧹 Limpiar datos huérfanos** - Elimina carpetas sin entrada en JSON
- **🔄 Resetear modelo** - Fuerza reentrenamiento cuando cambias señas

#### 💡 **Vista del Estado de Señas:**
```
📋 SEÑAS DISPONIBLES:
==================================================
 1. adios          | JSON: ✅ | DATA: ✅ | Archivos: 30
 2. como_estas     | JSON: ✅ | DATA: ✅ | Archivos: 30
 3. hola           | JSON: ✅ | DATA: ✅ | Archivos: 30
 4. nueva_seña     | JSON: ✅ | DATA: ❌ | Archivos: 0
```

#### ⚠️ **Importante después de cambios:**
Después de eliminar o agregar señas, siempre reentrenar:
```bash
python train_model.py
```

## ⚙️ Configuración Avanzada

### Ajustar Sensibilidad de Detección

En `detect_signs.py`, modificar:
```python
CONFIDENCE_THRESHOLD = 0.50  # 0.1-0.9 (más bajo = más sensible)
MIN_STABLE_FRAMES = 8        # 1-20 (más alto = más estable)
```

### Configurar Síntesis de Voz

El sistema detecta automáticamente voces en español. Para forzar una voz específica, modificar `voice_system.py`.

### Cambiar Cámara

En `detect_signs.py`:
```python
cap = cv2.VideoCapture(0)  # Cambiar 0 por 1, 2, etc.
```

## 🔧 Solución de Problemas

### Error: "No module named 'cv2'"
```bash
pip uninstall opencv-python
pip install opencv-python==4.10.0.84
```

### Error: "No se detecta la cámara"
1. Verificar que la cámara funciona en otras aplicaciones
2. Cambiar el índice de cámara: `cv2.VideoCapture(1)`
3. En Windows: verificar permisos de cámara

### Error: "MediaPipe no funciona"
```bash
pip uninstall mediapipe
pip install mediapipe==0.10.21
```

### Problemas de Audio/Voz
1. Verificar altavoces funcionando
2. En Windows: verificar permisos de micrófono
3. Instalar codecs de audio: `pip install pygame gTTS`

### Modelo no entrena correctamente
1. Verificar que cada seña tiene mínimo 10 secuencias válidas
2. Hacer señas más variadas y claras
3. Mejorar iluminación durante recolección

### TensorFlow muy lento
```bash
# Para CPU más rápida
set TF_ENABLE_ONEDNN_OPTS=0

# O instalar versión GPU (opcional)
pip install tensorflow-gpu==2.17.0
```

## 📊 Señas Incluidas por Defecto

El sistema viene con soporte para estas señas (puedes agregar más):
- 👋 **hola**
- 👋 **adios** 
- 🤔 **como**
- 😊 **como_estas**
- 😞 **mal**
- 🔢 **cuanto**
- 💭 **sientes**
fotos estas valen

## 🚀 Agregar Nuevas Señas

1. **Recolectar datos:**
   ```bash
   python collect_data.py
   ```

2. **Reentrenar modelo:**
   ```bash
   python train_model.py
   ```

3. **¡Listo!** El sistema automáticamente:
   - Actualiza `signs.json`
   - Genera audio para la nueva seña
   - La incluye en la detección

## 📈 Rendimiento Esperado

- **Precisión:** 85-95% con datos de calidad
- **Tiempo de respuesta:** <100ms por frame
- **Señas simultáneas:** Detecta mejor mano automáticamente
- **Requisitos mínimos:** 4GB RAM, CPU dual-core

## 🤝 Contribuir

1. Fork del repositorio
2. Crear rama para nueva característica
3. Commit con cambios
4. Push a la rama
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👨‍💻 Autor

**Josu-F1**
- GitHub: [@Josu-F1](https://github.com/Josu-F1)
- Proyecto: [SignLanguageDetection](https://github.com/Josu-F1/SignLanguageDetection)

## 🎯 Próximas Características

- [ ] Soporte para más idiomas de voz
- [ ] Detección de expresiones faciales
- [ ] Modo de entrenamiento supervisado
- [ ] API REST para integración
- [ ] Aplicación móvil
- [ ] Soporte para gestos complejos

---

¿Problemas? Crear un [Issue](https://github.com/Josu-F1/SignLanguageDetection/issues) en GitHub 🚀