# 🤟 Sistema de Reconocimiento de Lenguaje de Señas con Voz

Un sistema inteligente de reconocimiento de lenguaje de señas en tiempo real que detecta gestos de las manos usando cámara web y convierte las señas reconocidas a voz en español.

## 🚀 Características

- **Detección en tiempo real** de señas usando MediaPipe
- **Reconocimiento de ambas manos** con selección inteligente
- **Síntesis de voz en español** con sistema de audio robusto
- **Interfaz visual intuitiva** con información detallada
- **Sistema de entrenamiento mejorado** con validación de datos
- **Recolección de datos fácil** para nuevas señas
- **Gestor de señas integrado** para agregar/eliminar/renombrar palabras
- **Modelo LSTM profundo** para alta precisión

## 📋 Requisitos del Sistema

### Sistema Operativo
- **Windows 10/11** (recomendado)
- **macOS 10.14+** o **Linux Ubuntu 18.04+**

### Hardware
- **Cámara web** (resolución mínima 640x480)
- **Micrófono y altavoces** para síntesis de voz
- **4GB RAM** mínimo, 8GB recomendado
- **2GB espacio libre** en disco

### Software Base
- **Python 3.11.x** (IMPORTANTE: no usar 3.12+ ni 3.10-)
- **Git** (opcional, para clonar el repositorio)

## 🛠️ Instalación Completa

### Paso 1: Instalar Python 3.11

#### En Windows:
1. Descargar Python 3.11.9 desde: https://www.python.org/downloads/release/python-3119/
2. Durante la instalación marcar: ✅ "Add Python to PATH"
3. Verificar instalación:
```cmd
python --version
# Debe mostrar: Python 3.11.9
```

#### En macOS:
```bash
# Usando Homebrew
brew install python@3.11

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