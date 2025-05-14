import os
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO
from tqdm import tqdm
import sys

# Paramètres
IMG_SIZE = 224  # Taille d'entrée pour le modèle
BATCH_SIZE = 16
EPOCHS = 20

# Chemins des dossiers
TRAIN_DIR = 'Training_Set'
TEST_DIR = 'Test_Set'
DATASET_DIR = 'dataset'
MODEL_PATH = 'fish_classifier.pt'

def get_class_dict():
    """
    Crée un dictionnaire de mappage des classes à partir des dossiers existants.
    """
    # Vérifier que les dossiers contiennent des données
    train_classes = os.listdir(os.path.join(DATASET_DIR, 'train'))
    if not train_classes:
        raise ValueError(f"Aucune classe trouvée dans {os.path.join(DATASET_DIR, 'train')}. Vérifiez vos dossiers d'origine.")
    
    print(f"Dataset contient {len(train_classes)} classes")
    
    # Sauvegarder les noms des classes
    with open('class_names.txt', 'w') as f:
        for class_name in sorted(train_classes):
            f.write(f"{class_name}\n")
    
    # Créer un dictionnaire de mappage des classes
    class_dict = {i: class_name for i, class_name in enumerate(sorted(train_classes))}
    
    return class_dict

def train_model(class_dict):
    """
    Entraîne un modèle de classification en utilisant directement PyTorch.
    """
    print("Démarrage de l'entraînement...")
    
    # Désactiver les logs TensorBoard pour éviter les conflits avec TensorFlow
    os.environ["WANDB_DISABLED"] = "true"
    
    # Obtenir le chemin absolu du dossier dataset
    dataset_path = os.path.abspath(DATASET_DIR)
    
    try:
        # Importer les bibliothèques nécessaires pour PyTorch
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms, models
        
        print(f"Entraînement du modèle avec le dataset: {dataset_path}")
        
        # Définir les transformations pour les images
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        # Charger les datasets
        image_datasets = {
            'train': datasets.ImageFolder(os.path.join(dataset_path, 'train'), data_transforms['train']),
            'val': datasets.ImageFolder(os.path.join(dataset_path, 'val'), data_transforms['val'])
        }
        
        # Créer les dataloaders
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
            'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        }
        
        # Obtenir le nombre de classes
        num_classes = len(class_dict)
        print(f"Nombre de classes: {num_classes}")
        print(f"Nombre d'images d'entraînement: {len(image_datasets['train'])}")
        print(f"Nombre d'images de validation: {len(image_datasets['val'])}")
        
        # Vérifier si CUDA est disponible
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation de: {device}")
        
        # Charger un modèle pré-entraîné (ResNet18)
        model = models.resnet18(pretrained=True)
        
        # Remplacer la dernière couche pour notre classification
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Déplacer le modèle sur le GPU si disponible
        model = model.to(device)
        
        # Définir la fonction de perte et l'optimiseur
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # Réduire le taux d'apprentissage lorsque le plateau est atteint
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        
        # Créer le dossier pour sauvegarder les modèles
        os.makedirs('runs/fish_classifier/weights', exist_ok=True)
        
        # Entraîner le modèle
        best_acc = 0.0
        best_model_path = Path('runs/fish_classifier/weights/best.pt')
        
        for epoch in range(EPOCHS):
            print(f'Epoch {epoch+1}/{EPOCHS}')
            print('-' * 10)
            
            # Chaque epoch a une phase d'entraînement et de validation
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Mode entraînement
                else:
                    model.eval()   # Mode évaluation
                
                running_loss = 0.0
                running_corrects = 0
                
                # Itérer sur les données
                for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Mettre les gradients à zéro
                    optimizer.zero_grad()
                    
                    # Forward
                    # Activer le calcul des gradients seulement en phase d'entraînement
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Backward + optimize seulement en phase d'entraînement
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # Statistiques
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Si nous sommes en phase de validation, mettre à jour le scheduler
                if phase == 'val':
                    scheduler.step(epoch_loss)
                    
                    # Sauvegarder le meilleur modèle
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_path)
                        print(f'Meilleur modèle sauvegardé avec précision: {best_acc:.4f}')
        
        print(f'Meilleure précision de validation: {best_acc:.4f}')
        
        # Charger le meilleur modèle
        model.load_state_dict(torch.load(best_model_path))
        
        # Sauvegarder le modèle complet
        torch.save(model, MODEL_PATH)
        print(f"Modèle sauvegardé sous '{MODEL_PATH}'")
        
        # Créer un wrapper pour le modèle PyTorch qui imite l'API YOLO
        class PyTorchModelWrapper:
            def __init__(self, model, class_names, transform):
                self.model = model
                self.class_names = class_names
                self.transform = transform
                self.device = next(model.parameters()).device
            
            def predict(self, img_path, verbose=True):
                from PIL import Image
                import numpy as np
                
                class Result:
                    def __init__(self, probs):
                        self.probs = probs
                
                class Probs:
                    def __init__(self, top1, top5, data):
                        self.top1 = top1
                        self.top5 = top5
                        self.data = data
                
                # Charger et transformer l'image
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                # Prédire
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
                    # Obtenir les 5 meilleures prédictions
                    top5_probs, top5_indices = torch.topk(probs, 5)
                    
                    # Créer l'objet de résultat
                    probs_obj = Probs(
                        top1=top5_indices[0].item(),
                        top5=top5_indices.tolist(),
                        data=probs.cpu().numpy()
                    )
                    
                    return [Result(probs_obj)]
        
        # Créer le wrapper pour le modèle
        model_wrapper = PyTorchModelWrapper(
            model, 
            list(class_dict.values()),
            data_transforms['val']
        )
        
        return model_wrapper
        
    except Exception as e:
        print(f"Erreur lors de l'entraînement avec PyTorch: {e}")
        print("Tentative d'utiliser YOLO directement...")
        
        # Essayer avec YOLO comme dernier recours
        try:
            # Charger un modèle YOLOv8 pré-entraîné pour la classification
            model = YOLO('yolov8n-cls.pt')
            
            # Utiliser Python pour exécuter la commande ultralytics directement
            cmd = f"python -c \"from ultralytics import YOLO; model = YOLO('yolov8n-cls.pt'); model.train(data='{dataset_path}', epochs={EPOCHS}, imgsz={IMG_SIZE}, batch={BATCH_SIZE}, name='fish_classifier', project='runs', exist_ok=True)\""
            print(f"Exécution de la commande: {cmd}")
            os.system(cmd)
            
            # Vérifier si le modèle a été créé
            best_model_path = Path('runs/fish_classifier/weights/best.pt')
            if best_model_path.exists():
                shutil.copy2(best_model_path, MODEL_PATH)
                print(f"Meilleur modèle sauvegardé sous '{MODEL_PATH}'")
                return YOLO(MODEL_PATH)
            else:
                print(f"Modèle non trouvé à {best_model_path}")
                return model
        except Exception as e2:
            print(f"Deuxième erreur lors de l'entraînement: {e2}")
            return None

def evaluate_model(model, class_dict):
    """
    Évalue le modèle et génère un rapport de classification et une matrice de confusion.
    """
    print("Évaluation du modèle...")
    
    # Pour la matrice de confusion et le rapport de classification,
    # nous devons exécuter le modèle sur l'ensemble de validation et collecter les prédictions
    val_dir = os.path.join(DATASET_DIR, 'val')
    class_names = list(class_dict.values())
    
    # Créer un mappage inverse des noms de classe aux indices
    class_to_idx = {name: idx for idx, name in class_dict.items()}
    
    # Collecter les vraies étiquettes et les prédictions
    y_true = []
    y_pred = []
    
    # Évaluation manuelle
    print("Évaluation des prédictions...")
    
    # Limiter le nombre d'images à évaluer pour accélérer le processus
    max_images_per_class = 50
    total_images = 0
    
    for class_name in sorted(os.listdir(val_dir)):
        class_dir = os.path.join(val_dir, class_name)
        
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Évaluation de la classe {class_name}...")
        
        # Obtenir l'indice de classe correct
        class_idx = class_to_idx.get(class_name, -1)
        if class_idx == -1:
            print(f"Avertissement: Classe {class_name} non trouvée dans le dictionnaire de classes")
            continue
        
        # Obtenir la liste des fichiers image
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Limiter le nombre d'images
        image_files = image_files[:max_images_per_class]
        total_images += len(image_files)
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Prédire avec le modèle
                results = model.predict(img_path, verbose=False)
                pred_class = results[0].probs.top1
                
                # Vérifier que l'indice prédit est valide
                if pred_class < len(class_dict):
                    y_true.append(class_idx)
                    y_pred.append(pred_class)
                else:
                    print(f"Avertissement: Indice de classe prédit {pred_class} hors limites")
            except Exception as e:
                print(f"Erreur lors de la prédiction pour {img_path}: {e}")
    
    print(f"Évaluation terminée sur {total_images} images")
    
    if len(y_true) == 0:
        print("Aucune prédiction n'a pu être faite. Vérifiez les données et le modèle.")
        return
    
    # Vérifier que les dimensions correspondent
    if len(set(y_true)) != len(set(y_pred)):
        print(f"Avertissement: Nombre de classes différent entre y_true ({len(set(y_true))}) et y_pred ({len(set(y_pred))})")
        
    # Ajuster les noms de classes pour correspondre aux indices réellement utilisés
    used_classes = sorted(list(set(y_true)))
    used_class_names = [class_names[i] for i in used_classes if i < len(class_names)]
    
    # Générer le rapport de classification
    print("\nRapport de classification:")
    print(classification_report(y_true, y_pred, target_names=used_class_names))
    
    # Générer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=used_class_names, yticklabels=used_class_names)
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes')
    plt.title('Matrice de confusion')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def train_and_evaluate():
    """
    Fonction principale pour entraîner et évaluer le modèle.
    """
    # Obtenir le dictionnaire des classes
    class_dict = get_class_dict()
    
    # Entraîner le modèle
    model = train_model(class_dict)
    
    # Évaluer le modèle
    evaluate_model(model, class_dict)
    
    print("Entraînement et évaluation terminés.")

if __name__ == "__main__":
    train_and_evaluate()