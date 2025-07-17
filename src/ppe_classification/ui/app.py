import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import time
import os
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from ultralytics import YOLO

from ppe_classification.models.fish_classifier import FishClassifier
from ppe_classification.utils.fish_counter import FishCounter

class FishClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Système de Classification de Poissons")
        self.root.geometry("1280x720")
        
        # Variables
        self.is_running = False
        self.use_webcam = True
        self.video_path = None
        self.cap = None
        self.classifier = None
        self.detector = None  # Modèle YOLOv8 pour la détection
        self.counter = None
        self.frame_count = 0
        self.fps = 0
        self.start_time = 0
        
        # Créer l'interface
        self.create_ui()
        
        # Charger le modèle si disponible
        self.load_model()
    
    def create_ui(self):
        # Frame principale
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame gauche (vidéo et contrôles)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame vidéo
        self.video_frame = ttk.Frame(left_frame, relief=tk.SUNKEN, borderwidth=2)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Label pour afficher la vidéo
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Frame de contrôle
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Boutons de contrôle
        self.webcam_btn = ttk.Button(control_frame, text="Utiliser Webcam", command=self.use_webcam_source)
        self.webcam_btn.pack(side=tk.LEFT, padx=5)
        
        self.video_btn = ttk.Button(control_frame, text="Charger Vidéo", command=self.load_video)
        self.video_btn.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = ttk.Button(control_frame, text="Démarrer", command=self.start_processing)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Arrêter", command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = ttk.Button(control_frame, text="Réinitialiser Compteurs", command=self.reset_counters)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_csv_btn = ttk.Button(control_frame, text="Effacer CSV", command=self.clear_csv_data)
        self.clear_csv_btn.pack(side=tk.LEFT, padx=5)
        
        self.report_btn = ttk.Button(control_frame, text="Générer rapport", command=self.run_metrics_report)
        self.report_btn.pack(side=tk.LEFT, padx=5)
        
        # Seuil de confiance
        threshold_frame = ttk.Frame(left_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(threshold_frame, text="Seuil de confiance:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                                   variable=self.threshold_var, length=200)
        threshold_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        threshold_label = ttk.Label(threshold_frame, textvariable=tk.StringVar(value="0.5"))
        threshold_label.pack(side=tk.LEFT, padx=5)
        
        def update_threshold_label(*args):
            threshold_label.config(text=f"{self.threshold_var.get():.1f}")
        
        self.threshold_var.trace_add("write", update_threshold_label)
        
        # Frame droite (statistiques et résultats)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        # Onglets pour les statistiques
        tab_control = ttk.Notebook(right_frame)
        
        # Onglet de comptage
        count_tab = ttk.Frame(tab_control)
        tab_control.add(count_tab, text="Comptage")
        
        # Tableau de comptage
        self.count_tree = ttk.Treeview(count_tab, columns=("Espèce", "Comptage"), show="headings")
        self.count_tree.heading("Espèce", text="Espèce")
        self.count_tree.heading("Comptage", text="Comptage")
        self.count_tree.column("Espèce", width=150)
        self.count_tree.column("Comptage", width=100)
        self.count_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Label d'information
        self.info_label = ttk.Label(right_frame, text="FPS: 0.0 | Détections: 0")
        self.info_label.pack(pady=5)

    def load_model(self):
        try:
            # Définir les chemins vers les modèles
            models_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models')
            fish_classifier_path = os.path.join(models_dir, 'fish_classifier.pt')
            class_names_path = os.path.join(models_dir, 'class_names.txt')
            yolo_path = os.path.join(models_dir, 'yolov8n.pt')
            
            # Charger le modèle de classification
            if os.path.exists(fish_classifier_path):
                self.classifier = FishClassifier(fish_classifier_path, class_names_path, confidence_threshold=self.threshold_var.get())
                print("Modèle de classification chargé avec succès.")
            else:
                print("Modèle de classification non trouvé.")
                self.classifier = None
            
            # Charger le modèle de détection YOLOv8
            if os.path.exists(yolo_path):
                self.detector = YOLO(yolo_path)
                print("Modèle de détection YOLOv8 chargé avec succès.")
            else:
                # Télécharger le modèle YOLOv8 si non disponible
                print("Téléchargement du modèle de détection YOLOv8...")
                self.detector = YOLO('yolov8n.pt')  # Cela téléchargera automatiquement le modèle
            
            # Initialiser le compteur
            self.counter = FishCounter()
            
        except Exception as e:
            self.show_error("Erreur de chargement du modèle", str(e))
    
    def use_webcam_source(self):
        self.use_webcam = True
        print("Source vidéo: Webcam")
        
        # Tester l'ouverture de la webcam immédiatement pour informer l'utilisateur
        test_cap = cv2.VideoCapture(0)
        if not test_cap.isOpened():
            # Essayer avec d'autres indices de caméra
            for i in range(1, 3):  # Essayer les indices 1 et 2
                test_cap.release()
                test_cap = cv2.VideoCapture(i)
                if test_cap.isOpened():
                    self.webcam_index = i
                    print(f"Webcam trouvée à l'indice {i}")
                    test_cap.release()
                    return
            
            # Si aucune webcam n'est trouvée
            test_cap.release()
            self.show_error("Erreur Webcam", 
                           "Impossible d'accéder à la webcam. Vérifiez que :\n"
                           "1. Votre webcam est connectée\n"
                           "2. Aucune autre application n'utilise la webcam\n"
                           "3. Vous avez accordé les permissions d'accès à la webcam")
        else:
            self.webcam_index = 0
            test_cap.release()
    
    def load_video(self):
        video_path = filedialog.askopenfilename(
            title="Sélectionner une vidéo",
            filetypes=(("Fichiers vidéo", "*.mp4 *.avi *.mov"), ("Tous les fichiers", "*.*"))
        )
        if video_path:
            self.video_path = video_path
            self.use_webcam = False
            print(f"Source vidéo: {self.video_path}")
    
    def start_processing(self):
        if self.is_running:
            return
        
        # Vérifier si les modèles sont chargés
        if self.classifier is None and self.detector is None:
            self.show_error("Erreur", "Aucun modèle n'est chargé. Veuillez entraîner ou télécharger un modèle.")
            return
        
        # Ouvrir la source vidéo
        if self.use_webcam:
            # Utiliser l'indice de webcam trouvé précédemment ou l'indice par défaut
            webcam_index = getattr(self, 'webcam_index', 0)
            self.cap = cv2.VideoCapture(webcam_index)
            
            # Définir une résolution standard pour la webcam
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            if self.video_path:
                self.cap = cv2.VideoCapture(self.video_path)
            else:
                self.show_error("Erreur", "Aucune source vidéo sélectionnée.")
                return
        
        # Vérifier si la source vidéo est ouverte
        if not self.cap.isOpened():
            self.show_error("Erreur", "Impossible d'ouvrir la source vidéo.")
            return
        
        # Mettre à jour l'interface
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.webcam_btn.config(state=tk.DISABLED)
        self.video_btn.config(state=tk.DISABLED)
        
        # Démarrer le traitement
        self.is_running = True
        self.frame_count = 0
        self.start_time = time.time()
        
        # Lancer le thread de traitement
        self.process_thread = threading.Thread(target=self.process_video)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def stop_processing(self):
        if not self.is_running:
            return
        
        # Arrêter le traitement
        self.is_running = False
        
        # Mettre à jour l'interface
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.webcam_btn.config(state=tk.NORMAL)
        self.video_btn.config(state=tk.NORMAL)
        
        # Libérer la source vidéo
        if self.cap:
            self.cap.release()
    
    def reset_counters(self):
        if self.counter:
            self.counter.reset()
            self.update_count_display()
    
    def clear_csv_data(self):
        """Efface les données du fichier CSV"""
        if self.counter:
            self.counter.clear_csv()
            messagebox.showinfo("Information", "Le fichier CSV a été effacé avec succès.")
    
    def process_video(self):
        while self.is_running:
            # Lire une frame
            ret, frame = self.cap.read()
            
            if not ret:
                # Fin de la vidéo ou erreur
                if not self.use_webcam:
                    # Redémarrer la vidéo
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # Erreur de la webcam
                    self.root.after(0, lambda: self.show_error("Erreur", "Erreur de lecture de la webcam."))
                    self.root.after(0, self.stop_processing)
                    break
            
            # Mettre à jour le compteur de frames et calculer le FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                self.fps = self.frame_count / elapsed_time
            
            # Détecter les poissons
            detections = self.detect_fish(frame)
            
            # Mettre à jour le compteur
            tracked_fish = self.counter.update(detections, frame)
            
            # Dessiner les détections
            self.draw_detections(frame, detections, tracked_fish)
            
            # Créer une copie de la frame pour éviter les modifications pendant l'affichage
            display_frame = frame.copy()
            num_detections = len(detections)
            
            # Mettre à jour l'affichage en une seule fois pour éviter les redimensionnements multiples
            def update_all_displays():
                self.display_frame(display_frame)
                self.update_count_display()
                self.update_info_display(num_detections)
            
            # Programmer une seule mise à jour de l'interface
            self.root.after(1, update_all_displays)
            
            # Limiter le FPS pour ne pas surcharger l'interface
            time.sleep(0.03)  # Augmenter légèrement le délai pour réduire la charge CPU
    
    def detect_fish(self, frame):
        detections = []
        
        # Utiliser YOLOv8 pour détecter les objets
        results = self.detector(frame, conf=self.threshold_var.get(), verbose=False)
        
        # Filtrer les détections pour ne garder que les poissons (classe 1 dans COCO)
        # Note: YOLOv8 utilise l'indice 1 pour la classe "personne", mais nous pouvons détecter tous les objets
        # et les classifier ensuite avec notre modèle de classification
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extraire les coordonnées de la boîte
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                
                # Extraire la région d'intérêt (ROI)
                roi = frame[y:y+h, x:x+w]
                
                if roi.size == 0:
                    continue
                
                # Classifier la ROI si un classificateur est disponible
                if self.classifier:
                    species, confidence = self.classifier.classify(roi)
                else:
                    # Utiliser la classe prédite par YOLOv8 si pas de classificateur spécifique
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    species = result.names[class_id]
                
                if species:
                    bbox = (x, y, w, h)
                    detections.append((bbox, species, confidence))
        
        return detections
    
    def draw_detections(self, frame, detections, tracked_fish):
        for detection in detections:
            bbox, species, confidence = detection
            x, y, w, h = bbox
            
            # Dessiner le rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Afficher l'espèce et la confiance
            label = f"{species}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Dessiner les IDs des poissons suivis
        for fish_id, (centroid, species, confidence) in tracked_fish.items():
            cx, cy = centroid
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(fish_id), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def display_frame(self, frame):
        # Convertir l'image pour l'affichage Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Obtenir les dimensions originales de l'image
        img_width, img_height = img.size
        
        # Obtenir les dimensions du conteneur
        container_width = self.video_frame.winfo_width()
        container_height = self.video_frame.winfo_height()
        
        # Définir une taille maximale pour éviter que l'image ne prenne trop de place
        max_width = min(container_width, 800)  # Limiter à 800px de large ou à la taille du conteneur
        max_height = min(container_height, 600)  # Limiter à 600px de haut ou à la taille du conteneur
        
        # Si le conteneur est trop petit, utiliser une taille minimale
        if container_width < 10 or container_height < 10:
            max_width = 640
            max_height = 480
        
        # Calculer le ratio pour conserver les proportions
        width_ratio = max_width / img_width
        height_ratio = max_height / img_height
        ratio = min(width_ratio, height_ratio)
        
        # Calculer les nouvelles dimensions
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Redimensionner l'image
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convertir en PhotoImage
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Mettre à jour l'affichage
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk  # Garder une référence
    
    def update_count_display(self):
        # Mettre à jour l'affichage du comptage
        counts = self.counter.get_counts()
        
        # Effacer le tableau
        for item in self.count_tree.get_children():
            self.count_tree.delete(item)
        
        # Ajouter les nouvelles données
        for species, count in counts.items():
            self.count_tree.insert("", tk.END, values=(species, count))
    
    def update_info_display(self, num_detections):
        # Mettre à jour l'affichage des informations
        self.info_label.config(text=f"FPS: {self.fps:.1f} | Détections: {num_detections}")
    
    def show_error(self, title, message):
        # Afficher une boîte de dialogue d'erreur
        messagebox.showerror(title, message)
    
    def run_metrics_report(self):
        import subprocess
        import sys
        metrics_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'metrics.py')
        subprocess.Popen([sys.executable, metrics_path])
    
    def show_metrics(self):
        # Cette fonction serait appelée pour afficher les métriques après l'évaluation du modèle
        # Elle pourrait être liée à un bouton dans l'interface
        
        # Créer une figure pour la matrice de confusion
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Exemple de matrice de confusion (à remplacer par les vraies données)
        classes = list(self.counter.get_counts().keys())
        if not classes:
            classes = ["Pas de données"]
        
        cm = np.zeros((len(classes), len(classes)))
        
        # Afficher la matrice
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel('Prédictions')
        ax.set_ylabel('Vraies étiquettes')
        ax.set_title('Matrice de confusion')
        
        # Supprimer les widgets existants
        for widget in self.confusion_frame.winfo_children():
            widget.destroy()
        
        # Ajouter la figure à l'interface
        canvas = FigureCanvasTkAgg(fig, master=self.confusion_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = FishClassificationApp(root)
    root.mainloop()