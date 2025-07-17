import cv2
import numpy as np
import torch
from torchvision import transforms
import os

class FishClassifier:
    def __init__(self, model_path, class_names_path, confidence_threshold=0.7):
        # Charger le modèle PyTorch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()  # Mettre en mode évaluation
        
        # Charger les noms des classes
        with open(class_names_path, 'r', encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        self.img_size = 224  # Taille d'entrée du modèle
        self.confidence_threshold = confidence_threshold
        
        # Définir les transformations pour les images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """
        Prétraite l'image pour l'inférence avec PyTorch.
        """
        # Redimensionner l'image
        img = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convertir en RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convertir en tensor PyTorch
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Ajouter dimension batch
        img_tensor = img_tensor.to(self.device)
            
        return img_tensor
    
    def classify(self, image):
        """
        Classifie une image en utilisant le modèle PyTorch.
        
        Args:
            image: Image à classifier (format BGR de OpenCV)
            
        Returns:
            species: Nom de l'espèce prédite
            confidence: Score de confiance
        """
        # Prétraiter l'image
        img_tensor = self.preprocess_image(image)
        
        # Faire la prédiction avec PyTorch
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Obtenir l'indice de la classe avec la plus haute probabilité
            confidence, class_idx = torch.max(probs, 0)
            confidence = confidence.item()
            class_idx = class_idx.item()
        
        # Retourner la classe prédite et la confiance si elle dépasse le seuil
        if confidence >= self.confidence_threshold:
            species = self.class_names[class_idx]
            # Corriger le nom "muraine" en "Murène"
            if species.lower() == "muraine":
                species = "Murene"
            return species, confidence
        else:
            return None, confidence