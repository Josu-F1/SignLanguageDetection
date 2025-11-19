#!/bin/bash

# Script de instalación para Ubuntu 22.04
# Sistema de Detección de Lenguaje de Señas

echo "=========================================="
echo "Instalación de Dependencias"
echo "Ubuntu 22.04"
echo "=========================================="

# Actualizar repositorios
echo ""
echo "[1/6] Actualizando repositorios del sistema..."
sudo apt update

# Instalar Python 3.11 si no está instalado
echo ""
echo "[2/6] Verificando Python 3.11..."
if ! command -v python3.11 &> /dev/null
then
    echo "Instalando Python 3.11..."
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
else
    echo "Python 3.11 ya está instalado"
fi

# Instalar pip para Python 3.11
echo ""
echo "[3/6] Instalando pip para Python 3.11..."
sudo apt install -y python3-pip

# Instalar dependencias del sistema
echo ""
echo "[4/6] Instalando dependencias del sistema..."
sudo apt install -y \
    libopencv-dev \
    python3-opencv \
    libportaudio2 \
    portaudio19-dev \
    espeak \
    libespeak-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Crear entorno virtual (opcional)
echo ""
echo "[5/6] ¿Desea crear un entorno virtual? (recomendado) [s/n]"
read -r respuesta
if [[ "$respuesta" =~ ^[Ss]$ ]]; then
    echo "Creando entorno virtual..."
    python3.11 -m venv venv
    source venv/bin/activate
    echo "Entorno virtual activado"
fi

# Instalar paquetes de Python
echo ""
echo "[6/6] Instalando paquetes de Python..."
python3.11 -m pip install --upgrade pip

# Instalar todas las dependencias
python3.11 -m pip install \
    opencv-python==4.10.0.84 \
    mediapipe==0.10.21 \
    numpy==1.26.4 \
    tensorflow==2.18.0 \
    scikit-learn==1.6.0 \
    pyttsx3==2.98 \
    gtts==2.5.4 \
    pygame==2.6.1

echo ""
echo "=========================================="
echo "Instalación completada"
echo "=========================================="
echo ""
echo "Paquetes instalados:"
python3.11 -m pip list | grep -E "(opencv|mediapipe|numpy|tensorflow|scikit|pyttsx3|gtts|pygame)"
echo ""
echo "Para ejecutar el programa:"
if [[ "$respuesta" =~ ^[Ss]$ ]]; then
    echo "  source venv/bin/activate"
fi
echo "  python3.11 detect_signs.py"
echo ""
echo "Para recolectar datos:"
echo "  python3.11 collect_data.py"
echo ""
echo "Para entrenar el modelo:"
echo "  python3.11 train_model.py"
echo ""
