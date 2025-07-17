import cv2
import numpy as np
from collections import defaultdict
import supervision as sv
import csv
import os
from datetime import datetime
import time

class FishCounter:
    def __init__(self, max_disappeared=20, confidence_threshold=0.9):
        # Dictionnaire pour stocker les poissons suivis
        self.tracked_fish = {}
        # Compteur pour les IDs uniques
        self.next_fish_id = 0
        # Compteur par espèce
        self.species_count = defaultdict(int)
        # Nombre maximum de frames où un poisson peut disparaître avant d'être considéré comme parti
        self.max_disappeared = max_disappeared
        # Seuil de confiance pour compter un poisson
        self.confidence_threshold = confidence_threshold
        # Dictionnaire pour stocker les poissons disparus
        self.disappeared = {}
        # Ensemble pour stocker les IDs des poissons déjà comptés
        self.counted_fish_ids = set()
        # Nom du fichier CSV fixe pour l'exportation
        # Créer le dossier data s'il n'existe pas
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        self.csv_filename = os.path.join(data_dir, "fish_counts.csv")
        # Largeur et hauteur de l'image
        self.image_width = None
        self.image_height = None
        # Seuil de distance dynamique
        self.dynamic_distance_threshold = None
        # Derniers centroids pour chaque poisson
        self.last_centroids = {}  # Pour stocker les centroids précédents
        # Limite le nombre de points par poisson
        self.max_points_per_fish = 3  
        # Dernier temps de frame
        self.last_frame_time = 0
        # Intervalle minimum entre les points
        self.min_frame_interval = 0.5  

        # Initialiser le tracker ByteTrack de Supervision
        self.tracker = sv.ByteTrack()
        
        # Vérifier si le fichier existe déjà
        file_exists = os.path.isfile(self.csv_filename)
        
        # Créer ou ouvrir le fichier CSV
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Écrire les en-têtes seulement si le fichier est nouveau
            if not file_exists:
                writer.writerow(['ID', 'Espèce', 'Confiance', 'Haute_Confiance', 'Horodatage', 'Session'])
            
            # Écrire une ligne de séparation pour indiquer une nouvelle session
            writer.writerow(['---', f"Nouvelle session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", '---', '---', '---', '---'])
    
    def register(self, centroid, species, confidence):
        # Enregistrer un nouveau poisson
        fish_id = self.next_fish_id
        self.tracked_fish[fish_id] = (centroid, species, confidence)
        self.disappeared[fish_id] = 0
        
        # Ne compter le poisson que si la confiance dépasse le seuil
        # et s'il n'a pas déjà été compté
        if confidence >= self.confidence_threshold and fish_id not in self.counted_fish_ids:
            self.species_count[species] += 1
            self.counted_fish_ids.add(fish_id)
            
            # Exporter vers CSV
            self._export_to_csv(fish_id, species, confidence)
        
        self.next_fish_id += 1
        return fish_id
    
    def _export_to_csv(self, fish_id, species, confidence):
        """Exporter les informations du poisson vers un fichier CSV"""
        session_id = datetime.now().strftime('%Y%m%d_%H%M')
        # Déterminer si la confiance est élevée (>0.9)
        haute_confiance = 1 if confidence > 0.9 else 0
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                fish_id, 
                species, 
                f"{confidence:.4f}",
                haute_confiance,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                session_id
            ])
    
    def deregister(self, fish_id):
        # Supprimer un poisson du suivi
        del self.tracked_fish[fish_id]
        del self.disappeared[fish_id]
    
    def update(self, detections, frame):
        # Mettre à jour la taille de l'image si nécessaire
        if self.image_width is None or self.image_height is None:
            self.image_width = frame.shape[1]
            self.image_height = frame.shape[0]
            diagonal = np.sqrt(self.image_width**2 + self.image_height**2)
            self.dynamic_distance_threshold = diagonal * 0.1  # Réduire à 10% de la diagonale

        current_time = time.time()
        
        # Si aucune détection, incrémenter le compteur de disparition pour tous les poissons
        if len(detections) == 0:
            for fish_id in list(self.disappeared.keys()):
                self.disappeared[fish_id] += 1
                
                # Si un poisson a disparu pendant trop longtemps, le supprimer
                if self.disappeared[fish_id] > self.max_disappeared:
                    self.deregister(fish_id)
            
            return self.tracked_fish
        
        # Convertir les détections en format compatible avec le tracker
        detections_for_tracker = []
        
        for detection in detections:
            bbox, species, confidence = detection
            x, y, w, h = bbox
            centroid = (x + w // 2, y + h // 2)
            
            # Vérifier si ce poisson correspond à un poisson déjà suivi
            tracked = False
            best_match = None
            min_distance = float('inf')
            
            # Trouver le poisson le plus proche
            for fish_id, (existing_centroid, _, _) in self.tracked_fish.items():
                distance = np.sqrt((centroid[0] - existing_centroid[0])**2 + 
                                 (centroid[1] - existing_centroid[1])**2)
                
                if distance < self.dynamic_distance_threshold and distance < min_distance:
                    min_distance = distance
                    best_match = fish_id
            
            if best_match is not None:
                # Si on trouve un bon match, mettre à jour et réinitialiser le compteur de disparition
                self.tracked_fish[best_match] = (centroid, species, confidence)
                self.disappeared[best_match] = 0
                tracked = True
            
            # Si ce poisson n'est pas déjà suivi, l'enregistrer comme nouveau
            if not tracked:
                # Vérifier si on a déjà assez de points pour ce poisson
                if best_match is not None and best_match in self.last_centroids:
                    if len(self.last_centroids[best_match]) >= self.max_points_per_fish:
                        # Garder les points les plus récents
                        self.last_centroids[best_match] = self.last_centroids[best_match][-self.max_points_per_fish:]
                
                # Enregistrer le poisson
                fish_id = self.next_fish_id
                self.tracked_fish[fish_id] = (centroid, species, confidence)
                self.disappeared[fish_id] = 0
                
                # Ne compter le poisson que si la confiance dépasse le seuil
                if confidence >= self.confidence_threshold and fish_id not in self.counted_fish_ids:
                    self.species_count[species] += 1
                    self.counted_fish_ids.add(fish_id)
                    self._export_to_csv(fish_id, species, confidence)
                
                self.next_fish_id += 1
        
        # Mettre à jour les compteurs de disparition pour les poissons non détectés
        for fish_id in list(self.disappeared.keys()):
            if fish_id not in self.tracked_fish:
                self.disappeared[fish_id] += 1
                
                # Si un poisson a disparu pendant trop longtemps, le supprimer
                if self.disappeared[fish_id] > self.max_disappeared:
                    self.deregister(fish_id)
        
        return self.tracked_fish
    
    def get_counts(self):
        # Retourner le comptage par espèce
        return dict(self.species_count)
    
    def reset(self):
        # Réinitialiser tous les compteurs
        self.species_count = defaultdict(int)
        self.tracked_fish = {}
        self.disappeared = {}
        self.next_fish_id = 0
        self.counted_fish_ids = set()
    
    def clear_csv(self):
        """Efface le contenu du fichier CSV et réinitialise avec seulement les en-têtes"""
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Écrire uniquement les en-têtes
            writer.writerow(['ID', 'Espèce', 'Confiance', 'Haute_Confiance', 'Horodatage', 'Session'])
            # Écrire une ligne de séparation pour indiquer une nouvelle session
            writer.writerow(['---', f"Nouvelle session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", '---', '---', '---', '---'])