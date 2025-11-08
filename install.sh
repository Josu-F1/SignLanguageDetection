#!/bin/bash

echo "================================================"
echo "  INSTALACION AUTOMATICA - Sistema de Señas"  
echo "================================================"
echo

# Verificar que Python este instalado
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 no encontrado"
    echo
    echo "📋 Instalar Python 3.11:"
    echo "   macOS: brew install python@3.11"
    echo "   Ubuntu: sudo apt install python3.11 python3.11-pip"
    echo
    exit 1
fi

echo "✅ Python encontrado:"
python3.11 --version
echo

# Actualizar pip
echo "🔄 Actualizando pip..."
python3.11 -m pip install --upgrade pip

# Instalar dependencias
echo
echo "📦 Instalando dependencias del proyecto..."
echo "   Esto puede tomar varios minutos..."
echo

python3.11 -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo
    echo "❌ Error durante la instalacion"
    echo
    echo "🔧 Intentando instalacion manual..."
    
    # Instalacion manual de las librerías críticas
    python3.11 -m pip install opencv-python==4.10.0.84
    python3.11 -m pip install mediapipe==0.10.21
    python3.11 -m pip install tensorflow==2.17.0
    python3.11 -m pip install keras==3.12.0
    python3.11 -m pip install numpy==1.26.4
    python3.11 -m pip install scikit-learn==1.7.2
    python3.11 -m pip install pyttsx3==2.99
    python3.11 -m pip install gTTS==2.5.4
    python3.11 -m pip install pygame==2.6.1
    python3.11 -m pip install matplotlib==3.10.7
    python3.11 -m pip install seaborn==0.13.2
    python3.11 -m pip install pandas==2.3.3
    python3.11 -m pip install scipy==1.16.3
fi

echo
echo "🧪 Verificando instalacion..."
python3.11 check_installation.py

if [ $? -eq 0 ]; then
    echo
    echo "================================================"
    echo "  🎉 ¡INSTALACION COMPLETADA EXITOSAMENTE!"
    echo "================================================"
    echo
    echo "📋 Comandos disponibles:"
    echo "   python3.11 collect_data.py    - Recopilar datos"
    echo "   python3.11 train_model.py     - Entrenar modelo"
    echo "   python3.11 detect_signs.py    - Detectar señas"
    echo
    echo "📚 Ver README.md para más información"
    echo
else
    echo
    echo "❌ Instalacion incompleta - revisar errores arriba"
    echo "📚 Consultar README.md para solución manual"
    echo
fi