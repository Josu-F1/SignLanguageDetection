#!/usr/bin/env python3
# Script para hacer backup del proyecto

import os
import shutil
import json
from datetime import datetime

def backup_project():
    """Crea un backup completo del proyecto"""
    
    # Crear carpeta de backup con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_{timestamp}"
    
    print(f"[BACKUP] Creando backup en: {backup_dir}")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Lista de archivos/carpetas importantes
    items_to_backup = [
        'data/',                          # Datos de entrenamiento
        'sign_language_model.keras',      # Modelo entrenado
        'signs.json',                     # Mapeo de señas
        'training_stats.json',            # Estadísticas
        'detect_signs.py',                # Script principal
        'train_model.py',                 # Script de entrenamiento
        'collect_data.py',                # Recolección de datos
        'voice_system.py',                # Sistema de voz
        'logs/'                           # Logs de TensorBoard
    ]
    
    # Hacer backup de cada item
    for item in items_to_backup:
        if os.path.exists(item):
            if os.path.isdir(item):
                print(f"📁 Copiando directorio: {item}")
                shutil.copytree(item, os.path.join(backup_dir, item))
            else:
                print(f"📄 Copiando archivo: {item}")
                shutil.copy2(item, backup_dir)
        else:
            print(f"[ADVERTENCIA] No encontrado: {item}")
    
    # Crear resumen del backup
    summary = {
        'backup_date': datetime.now().isoformat(),
        'items_backed_up': [item for item in items_to_backup if os.path.exists(item)],
        'model_exists': os.path.exists('sign_language_model.keras'),
        'data_folders': [],
        'total_sequences': 0
    }
    
    # Analizar datos si existen
    if os.path.exists('data'):
        for sign_folder in os.listdir('data'):
            sign_path = os.path.join('data', sign_folder)
            if os.path.isdir(sign_path):
                sequences = len([f for f in os.listdir(sign_path) if f.endswith('.npy')])
                summary['data_folders'].append({
                    'sign': sign_folder,
                    'sequences': sequences
                })
                summary['total_sequences'] += sequences
    
    # Guardar resumen
    with open(os.path.join(backup_dir, 'backup_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n[COMPLETADO] Backup completado!")
    print(f"[RESUMEN] Resumen:")
    print(f"  - Carpetas de señas: {len(summary['data_folders'])}")
    print(f"  - Total secuencias: {summary['total_sequences']}")
    print(f"  - Modelo presente: {'[SI]' if summary['model_exists'] else '[NO]'}")
    print(f"  - Ubicación: {backup_dir}/")
    
    return backup_dir

def restore_from_backup(backup_dir):
    """Restaura desde un backup específico"""
    if not os.path.exists(backup_dir):
        print(f"[ERROR] Backup no encontrado: {backup_dir}")
        return False
    
    print(f"[RESTAURAR] Restaurando desde: {backup_dir}")
    
    # Restaurar cada item
    for item in os.listdir(backup_dir):
        if item == 'backup_summary.json':
            continue
            
        backup_path = os.path.join(backup_dir, item)
        
        if os.path.isdir(backup_path):
            if os.path.exists(item):
                print(f"[LIMPIAR] Eliminando directorio existente: {item}")
                shutil.rmtree(item)
            print(f"📁 Restaurando directorio: {item}")
            shutil.copytree(backup_path, item)
        else:
            print(f"📄 Restaurando archivo: {item}")
            shutil.copy2(backup_path, item)
    
    print(f"[COMPLETADO] Restauracion completada desde {backup_dir}")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'restore':
        if len(sys.argv) > 2:
            restore_from_backup(sys.argv[2])
        else:
            # Mostrar backups disponibles
            backups = [d for d in os.listdir('.') if d.startswith('backup_')]
            if backups:
                print("📦 Backups disponibles:")
                for backup in sorted(backups, reverse=True):
                    print(f"  - {backup}")
                print(f"\nUso: python backup_project.py restore {backups[0]}")
            else:
                print("[INFO] No hay backups disponibles")
    else:
        backup_dir = backup_project()
        print(f"\n[INFO] Para restaurar mas tarde:")
        print(f"   python backup_project.py restore {backup_dir}")