#!/usr/bin/env python3
"""
Gestor de Señas - Sistema de Reconocimiento de Lenguaje de Señas
Permite agregar, eliminar, renombrar y limpiar señas del proyecto
"""

import os
import shutil
import json
from pathlib import Path

class SignManager:
    def __init__(self):
        self.data_dir = Path('data')
        self.signs_file = Path('signs.json')
        self.model_file = Path('sign_language_model.keras')
        
    def load_signs(self):
        """Carga las señas desde el archivo JSON"""
        if self.signs_file.exists():
            with open(self.signs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_signs(self, signs_dict):
        """Guarda las señas en el archivo JSON"""
        with open(self.signs_file, 'w', encoding='utf-8') as f:
            json.dump(signs_dict, f, indent=2, ensure_ascii=False)
    
    def list_signs(self):
        """Lista todas las señas disponibles"""
        print("\n📋 SEÑAS DISPONIBLES:")
        print("=" * 50)
        
        # Señas en signs.json
        signs = self.load_signs()
        json_signs = set(signs.values()) if signs else set()
        
        # Carpetas en data/
        data_signs = set()
        if self.data_dir.exists():
            data_signs = {folder.name for folder in self.data_dir.iterdir() if folder.is_dir()}
        
        # Mostrar información completa
        all_signs = json_signs.union(data_signs)
        
        if not all_signs:
            print("❌ No hay señas disponibles")
            return
            
        for i, sign in enumerate(sorted(all_signs), 1):
            json_status = "✅" if sign in json_signs else "❌"
            data_status = "✅" if sign in data_signs else "❌"
            
            data_count = 0
            if sign in data_signs:
                sign_path = self.data_dir / sign
                if sign_path.exists():
                    data_count = len([f for f in sign_path.iterdir() if f.suffix == '.npy'])
            
            print(f"{i:2}. {sign:15} | JSON: {json_status} | DATA: {data_status} | Archivos: {data_count}")
    
    def delete_sign(self, sign_name):
        """Elimina completamente una seña"""
        print(f"\n🗑️ Eliminando seña: '{sign_name}'")
        
        # Eliminar carpeta de datos
        sign_path = self.data_dir / sign_name
        if sign_path.exists():
            shutil.rmtree(sign_path)
            print(f"✅ Carpeta de datos eliminada: {sign_path}")
        else:
            print(f"⚠️ No se encontró carpeta de datos: {sign_path}")
        
        # Actualizar signs.json
        signs = self.load_signs()
        keys_to_remove = [k for k, v in signs.items() if v == sign_name]
        
        for key in keys_to_remove:
            del signs[key]
            print(f"✅ Eliminado del JSON: {key} -> {sign_name}")
        
        # Reindexar signs.json
        if signs:
            new_signs = {}
            for i, (old_key, value) in enumerate(sorted(signs.items(), key=lambda x: int(x[0]))):
                new_signs[str(i)] = value
            self.save_signs(new_signs)
            print("✅ JSON reindexado correctamente")
        else:
            self.save_signs({})
            print("✅ JSON limpiado (vacío)")
        
        print(f"🎉 Seña '{sign_name}' eliminada completamente")
    
    def rename_sign(self, old_name, new_name):
        """Renombra una seña"""
        print(f"\n✏️ Renombrando: '{old_name}' → '{new_name}'")
        
        # Renombrar carpeta de datos
        old_path = self.data_dir / old_name
        new_path = self.data_dir / new_name
        
        if old_path.exists():
            old_path.rename(new_path)
            print(f"✅ Carpeta renombrada: {old_path} → {new_path}")
        else:
            print(f"⚠️ No se encontró carpeta: {old_path}")
        
        # Actualizar signs.json
        signs = self.load_signs()
        for key, value in signs.items():
            if value == old_name:
                signs[key] = new_name
                print(f"✅ JSON actualizado: {key} -> '{new_name}'")
        
        self.save_signs(signs)
        print(f"🎉 Seña renombrada exitosamente")
    
    def clean_orphaned_data(self):
        """Limpia datos huérfanos (carpetas sin entrada en JSON)"""
        print("\n🧹 Limpiando datos huérfanos...")
        
        signs = self.load_signs()
        json_signs = set(signs.values()) if signs else set()
        
        if not self.data_dir.exists():
            print("❌ No existe carpeta 'data'")
            return
        
        orphaned = []
        for folder in self.data_dir.iterdir():
            if folder.is_dir() and folder.name not in json_signs:
                orphaned.append(folder.name)
        
        if not orphaned:
            print("✅ No hay datos huérfanos")
            return
        
        print(f"🗑️ Encontradas {len(orphaned)} carpetas huérfanas:")
        for folder in orphaned:
            print(f"  - {folder}")
        
        confirm = input("\n¿Eliminar todas las carpetas huérfanas? (s/N): ").lower()
        if confirm == 's':
            for folder_name in orphaned:
                folder_path = self.data_dir / folder_name
                shutil.rmtree(folder_path)
                print(f"✅ Eliminada: {folder_name}")
            print("🎉 Limpieza completada")
        else:
            print("❌ Operación cancelada")
    
    def reset_model(self):
        """Elimina el modelo para forzar reentrenamiento"""
        print("\n🔄 Reseteando modelo...")
        
        if self.model_file.exists():
            confirm = input("⚠️ ¿Eliminar modelo actual? Tendrás que reentrenar (s/N): ").lower()
            if confirm == 's':
                self.model_file.unlink()
                print("✅ Modelo eliminado. Ejecuta train_model.py para reentrenar")
            else:
                print("❌ Operación cancelada")
        else:
            print("✅ No hay modelo que eliminar")
    
    def add_sign_manually(self, sign_name):
        """Agrega una seña manualmente al JSON (sin datos)"""
        print(f"\n➕ Agregando seña: '{sign_name}'")
        
        signs = self.load_signs()
        
        # Verificar si ya existe
        if sign_name in signs.values():
            print(f"⚠️ La seña '{sign_name}' ya existe")
            return
        
        # Encontrar el siguiente índice
        if signs:
            next_index = str(max(int(k) for k in signs.keys()) + 1)
        else:
            next_index = "0"
        
        # Agregar al JSON
        signs[next_index] = sign_name
        self.save_signs(signs)
        
        print(f"✅ Seña '{sign_name}' agregada al JSON con índice {next_index}")
        print(f"💡 Usa collect_data.py para recopilar datos para esta seña")
    
    def interactive_menu(self):
        """Menú interactivo principal"""
        while True:
            print("\n" + "="*60)
            print("🎯 GESTOR DE SEÑAS - Sistema de Lenguaje de Señas")
            print("="*60)
            
            self.list_signs()
            
            print("\n🛠️ OPCIONES DISPONIBLES:")
            print("1. 📋 Listar señas")
            print("2. 🗑️ Eliminar seña")
            print("3. ✏️ Renombrar seña")
            print("4. ➕ Agregar nueva seña")
            print("5. 🧹 Limpiar datos huérfanos")
            print("6. 🔄 Resetear modelo (forzar reentrenamiento)")
            print("7. 🚪 Salir")
            
            choice = input("\n➤ Selecciona una opción (1-7): ").strip()
            
            if choice == '1':
                continue  # Ya se muestra la lista arriba
                
            elif choice == '2':
                sign_name = input("📝 Nombre de la seña a eliminar: ").strip()
                if sign_name:
                    confirm = input(f"⚠️ ¿Eliminar '{sign_name}' permanentemente? (s/N): ").lower()
                    if confirm == 's':
                        self.delete_sign(sign_name)
                    else:
                        print("❌ Operación cancelada")
                
            elif choice == '3':
                old_name = input("📝 Nombre actual de la seña: ").strip()
                new_name = input("📝 Nuevo nombre: ").strip()
                if old_name and new_name:
                    self.rename_sign(old_name, new_name)
            
            elif choice == '4':
                sign_name = input("📝 Nombre de la nueva seña: ").strip()
                if sign_name:
                    self.add_sign_manually(sign_name)
                    
            elif choice == '5':
                self.clean_orphaned_data()
                
            elif choice == '6':
                self.reset_model()
                
            elif choice == '7':
                print("👋 ¡Hasta luego!")
                break
                
            else:
                print("❌ Opción inválida")
            
            input("\n⏸️ Presiona ENTER para continuar...")

def main():
    """Función principal"""
    manager = SignManager()
    
    # Verificar si estamos en el directorio correcto
    if not Path('collect_data.py').exists():
        print("❌ Error: Ejecuta este script desde la carpeta del proyecto")
        print("   (debe contener collect_data.py, train_model.py, etc.)")
        return
    
    print("🎯 Gestor de Señas - Sistema de Lenguaje de Señas")
    print("Este script te ayuda a gestionar las señas del proyecto")
    
    manager.interactive_menu()

if __name__ == "__main__":
    main()