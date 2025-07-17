#!/usr/bin/env python3
"""
Script d'entrée principal pour l'application PPE Classification
"""

import sys
import os

# Ajouter le dossier src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Lance l'application principale"""
    try:
        import tkinter as tk
        from ppe_classification.ui.app import FishClassificationApp
        
        root = tk.Tk()
        app = FishClassificationApp(root)
        root.mainloop()
        
    except ImportError as e:
        print(f"Erreur d'importation: {e}")
        print("Assurez-vous que toutes les dépendances sont installées avec 'uv sync'")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors du lancement de l'application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()