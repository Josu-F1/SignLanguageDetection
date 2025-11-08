#!/usr/bin/env python3
"""
Script de verificación de instalación para el Sistema de Reconocimiento de Señas
Verifica que todas las dependencias estén instaladas correctamente
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    """Verifica que la versión de Python sea compatible"""
    version = sys.version_info
    print(f"🐍 Python versión: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 11:
        print("✅ Versión de Python compatible")
        return True
    else:
        print("❌ Se requiere Python 3.11.x")
        print("   Descargar desde: https://www.python.org/downloads/release/python-3119/")
        return False

def check_library(lib_name, import_name=None, version_attr=None):
    """Verifica si una librería está instalada y su versión"""
    if import_name is None:
        import_name = lib_name
    
    try:
        module = __import__(import_name)
        
        # Intentar obtener la versión
        version = "desconocida"
        if version_attr and hasattr(module, version_attr):
            version = getattr(module, version_attr)
        elif hasattr(module, '__version__'):
            version = module.__version__
        
        print(f"✅ {lib_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {lib_name}: NO INSTALADA")
        return False

def check_system_requirements():
    """Verifica requisitos del sistema"""
    print("\n🖥️ Verificando requisitos del sistema...")
    
    # Verificar cámara
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Cámara web: Disponible")
            cap.release()
        else:
            print("⚠️ Cámara web: No detectada o en uso")
    except:
        print("❌ No se puede verificar la cámara")
    
    # Verificar audio (Windows)
    try:
        if sys.platform == "win32":
            import winsound
            print("✅ Sistema de audio: Disponible (Windows)")
        else:
            print("ℹ️ Sistema de audio: No verificado en este OS")
    except:
        print("⚠️ Sistema de audio: No verificado")

def main():
    """Función principal de verificación"""
    print("🚀 VERIFICACIÓN DE INSTALACIÓN - Sistema de Reconocimiento de Señas")
    print("=" * 70)
    
    # Verificar versión de Python
    if not check_python_version():
        print("\n❌ Instalación no válida: Versión de Python incorrecta")
        return False
    
    print("\n📚 Verificando librerías principales...")
    
    # Lista de librerías críticas
    critical_libs = [
        ("OpenCV", "cv2", "__version__"),
        ("MediaPipe", "mediapipe", "__version__"),
        ("TensorFlow", "tensorflow", "__version__"),
        ("Keras", "keras", "__version__"),
        ("NumPy", "numpy", "__version__"),
        ("Scikit-learn", "sklearn", "__version__"),
    ]
    
    all_critical_ok = True
    for lib_name, import_name, version_attr in critical_libs:
        if not check_library(lib_name, import_name, version_attr):
            all_critical_ok = False
    
    print("\n🔊 Verificando librerías de voz...")
    
    voice_libs = [
        ("pyttsx3", "pyttsx3", "__version__"),
        ("gTTS", "gtts", "__version__"),
        ("pygame", "pygame", "version.ver"),
    ]
    
    voice_ok = True
    for lib_name, import_name, version_attr in voice_libs:
        if not check_library(lib_name, import_name, version_attr):
            voice_ok = False
    
    print("\n📊 Verificando librerías de análisis...")
    
    analysis_libs = [
        ("Matplotlib", "matplotlib", "__version__"),
        ("Seaborn", "seaborn", "__version__"),
        ("Pandas", "pandas", "__version__"),
        ("SciPy", "scipy", "__version__"),
    ]
    
    analysis_ok = True
    for lib_name, import_name, version_attr in analysis_libs:
        if not check_library(lib_name, import_name, version_attr):
            analysis_ok = False
    
    # Verificar requisitos del sistema
    check_system_requirements()
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📋 RESUMEN DE VERIFICACIÓN:")
    
    if all_critical_ok:
        print("✅ Librerías críticas: TODAS INSTALADAS")
    else:
        print("❌ Librerías críticas: FALTAN ALGUNAS")
    
    if voice_ok:
        print("✅ Sistema de voz: FUNCIONANDO")
    else:
        print("❌ Sistema de voz: PROBLEMAS DETECTADOS")
    
    if analysis_ok:
        print("✅ Herramientas de análisis: DISPONIBLES")
    else:
        print("⚠️ Herramientas de análisis: ALGUNAS FALTANTES")
    
    # Verificar archivos del proyecto
    print("\n📁 Verificando archivos del proyecto...")
    
    required_files = [
        "detect_signs.py",
        "collect_data.py", 
        "train_model.py",
        "voice_system.py"
    ]
    
    import os
    files_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}: NO ENCONTRADO")
            files_ok = False
    
    # Resultado final
    print("\n" + "=" * 70)
    
    if all_critical_ok and voice_ok and files_ok:
        print("🎉 ¡INSTALACIÓN COMPLETA Y LISTA PARA USAR!")
        print("\n📋 Próximos pasos:")
        print("1. Recopilar datos: python collect_data.py")
        print("2. Entrenar modelo: python train_model.py") 
        print("3. Detectar señas: python detect_signs.py")
    else:
        print("❌ INSTALACIÓN INCOMPLETA")
        print("\n🔧 Para instalar dependencias faltantes:")
        print("   pip install -r requirements.txt")
        print("\n📚 Ver README.md para instrucciones detalladas")
    
    return all_critical_ok and voice_ok and files_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)