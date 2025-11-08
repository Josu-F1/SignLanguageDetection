@echo off
echo ================================================
echo  INSTALACION AUTOMATICA - Sistema de Señas 
echo ================================================
echo.

REM Verificar que Python este instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python no esta instalado o no esta en PATH
    echo.
    echo 📋 Por favor instalar Python 3.11 desde:
    echo    https://www.python.org/downloads/release/python-3119/
    echo.
    echo ⚠️  IMPORTANTE: Marcar "Add Python to PATH" durante instalacion
    pause
    exit /b 1
)

echo ✅ Python encontrado:
python --version
echo.

REM Actualizar pip
echo 🔄 Actualizando pip...
python -m pip install --upgrade pip

REM Instalar dependencias
echo.
echo 📦 Instalando dependencias del proyecto...
echo    Esto puede tomar varios minutos...
echo.

python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ❌ Error durante la instalacion
    echo.
    echo 🔧 Intentando instalacion manual...
    
    REM Instalacion manual de las librerías críticas
    python -m pip install opencv-python==4.10.0.84
    python -m pip install mediapipe==0.10.21
    python -m pip install tensorflow==2.17.0
    python -m pip install keras==3.12.0
    python -m pip install numpy==1.26.4
    python -m pip install scikit-learn==1.7.2
    python -m pip install pyttsx3==2.99
    python -m pip install gTTS==2.5.4
    python -m pip install pygame==2.6.1
    python -m pip install matplotlib==3.10.7
    python -m pip install seaborn==0.13.2
    python -m pip install pandas==2.3.3
    python -m pip install scipy==1.16.3
)

echo.
echo 🧪 Verificando instalacion...
python check_installation.py

if %errorlevel% equ 0 (
    echo.
    echo ================================================
    echo  🎉 ¡INSTALACION COMPLETADA EXITOSAMENTE! 
    echo ================================================
    echo.
    echo 📋 Comandos disponibles:
    echo    python collect_data.py    - Recopilar datos
    echo    python train_model.py     - Entrenar modelo  
    echo    python detect_signs.py    - Detectar señas
    echo.
    echo 📚 Ver README.md para más información
    echo.
) else (
    echo.
    echo ❌ Instalacion incompleta - revisar errores arriba
    echo 📚 Consultar README.md para solución manual
    echo.
)

pause