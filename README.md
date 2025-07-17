# PPE Classification - Système de Classification et Comptage de Poissons

Ce projet propose une solution complète pour la détection, la classification et le comptage automatique de poissons dans des vidéos sous-marines, à l'aide de modèles d'intelligence artificielle (YOLOv8 pour la détection, ResNet18 pour la classification).

## Fonctionnalités principales

- **Détection et classification** de poissons parmi 22 espèces méditerranéennes
- **Comptage automatique** et suivi des individus pour éviter le double comptage
- **Interface graphique** intuitive permettant :
  - Lancement/arrêt de l'analyse vidéo (webcam ou fichier)
  - Visualisation en temps réel des détections et classifications
  - Affichage du comptage par espèce
  - Réglage du seuil de confiance
  - Export des résultats au format CSV
  - Génération de rapports graphiques

## Espèces supportées

| Espèce | Nom scientifique | Famille |
|--------|------------------|---------|
| Anthias | Anthias anthias | Serranidae |
| Atherine | Atherina hepsetus | Atherinidae |
| Bar européen | Dicentrarchus labrax | Moronidae |
| Bogue | Boops boops | Sparidae |
| Carangue | Trachurus trachurus | Carangidae |
| Daurade Royale | Sparus aurata | Sparidae |
| Daurade rose | Pagellus bogaraveo | Sparidae |
| Éperlan | Osmerus eperlanus | Osmeridae |
| Girelle | Coris julis | Labridae |
| Gobie | Gobius niger | Gobiidae |
| Grande raie pastenague | Dasyatis pastinaca | Dasyatidae |
| Grande vive | Trachinus draco | Trachinidae |
| Grondin | Chelidonichthys lucerna | Triglidae |
| Maquereau | Scomber scombrus | Scombridae |
| Mérou | Epinephelus marginatus | Serranidae |
| Mostelle | Phycis phycis | Phycidae |
| Mulet cabot | Mugil cephalus | Mugilidae |
| Murène | Muraena helena | Muraenidae |
| Orphie | Belone belone | Belonidae |
| Poisson scorpion | Scorpaena scrofa | Scorpaenidae |
| Rouget | Mullus surmuletus | Mullidae |
| Sole commune | Solea solea | Soleidae |

## Installation

### Prérequis

- **Python 3.11 ou 3.12** (Python 3.13 n'est pas encore supporté)
- **uv** (gestionnaire de paquets Python moderne)
- **Git** pour cloner le dépôt

### Installation d'uv

Si vous n'avez pas encore installé uv, suivez les instructions ci-dessous :

#### macOS et Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation du projet

1. **Cloner le dépôt**
   ```bash
   git clone <url-du-depot>
   cd PPE-Classification
   ```

2. **Installer les dépendances**
   ```bash
   uv sync
   ```
   
   Cette commande va :
   - Créer automatiquement un environnement virtuel
   - Installer toutes les dépendances nécessaires
   - Configurer Python 3.12 (requis pour la compatibilité tkinter)

3. **Vérifier l'installation**
   ```bash
   uv run python --version
   ```
   Vous devriez voir `Python 3.12.x`

## Utilisation

### Lancement de l'application

```bash
uv run run_app.py
```

### Interface graphique

1. **Choisir la source vidéo** :
   - **Webcam** : Cliquez sur "Utiliser Webcam"
   - **Fichier vidéo** : Cliquez sur "Charger Vidéo" et sélectionnez votre fichier

2. **Configurer les paramètres** :
   - Ajustez le seuil de confiance avec le curseur (0.1 à 1.0)
   - Un seuil plus élevé = détections plus précises mais moins nombreuses

3. **Démarrer l'analyse** :
   - Cliquez sur "Démarrer" pour lancer la détection
   - Observez les détections en temps réel
   - Le comptage par espèce s'affiche dans l'onglet "Comptage"

4. **Contrôler l'analyse** :
   - **Arrêter** : Pause l'analyse
   - **Réinitialiser Compteurs** : Remet les compteurs à zéro
   - **Effacer CSV** : Supprime les données exportées
   - **Générer rapport** : Lance l'analyse des performances

### Gestion des données

Les résultats sont automatiquement sauvegardés dans :
- `data/fish_counts.csv` : Comptage par espèce avec timestamps
- `assets/` : Graphiques et visualisations générés

## Architecture du projet

```
PPE-Classification/
├── src/
│   └── ppe_classification/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── fish_classifier.py      # Modèle de classification
│       ├── ui/
│       │   ├── __init__.py
│       │   └── app.py                  # Interface graphique
│       └── utils/
│           ├── __init__.py
│           ├── fish_counter.py         # Comptage et suivi
│           ├── metrics.py              # Métriques et performances
│           └── performances.py         # Analyses de performance
├── models/
│   ├── fish_classifier.pt             # Modèle de classification PyTorch
│   ├── class_names.txt                # Noms des espèces
│   └── yolov8n.pt                     # Modèle de détection YOLOv8
├── data/
│   ├── fish_counts.csv                # Données de comptage
│   └── temp_video.mp4                 # Vidéos temporaires
├── config/
│   ├── dataset.yaml                   # Configuration du dataset
│   └── train_config.yaml              # Configuration d'entraînement
├── assets/
│   └── *.png                          # Graphiques et visualisations
├── docs/                              # Documentation
├── run_app.py                         # Script de lancement principal
├── pyproject.toml                     # Configuration du projet
├── uv.lock                            # Versions exactes des dépendances
└── README.md                          # Ce fichier
```

## Développement

### Commandes utiles

```bash
# Installer de nouvelles dépendances
uv add <package-name>

# Mettre à jour les dépendances
uv sync --upgrade

# Exécuter des scripts
uv run <script.py>

# Lancer l'application en mode développement
uv run --reload run_app.py
```

### Personnalisation

- **Ajouter de nouvelles espèces** : Modifiez `models/class_names.txt`
- **Changer les seuils** : Ajustez les paramètres dans `src/ppe_classification/models/fish_classifier.py`
- **Modifier l'interface** : Éditez `src/ppe_classification/ui/app.py`

## Démonstration

Une démonstration vidéo du système est disponible ici : [Lien Google Drive](https://drive.google.com/file/d/1ctbgBSiCJYhsyPaWc5_azhEsEapO8-NO/view?usp=sharing)

## Résolution des problèmes

### Problèmes courants

1. **Erreur tkinter** :
   ```bash
   # Sur macOS
   brew install python-tk@3.12
   
   # Recréer l'environnement
   rm -rf .venv
   uv sync
   ```

2. **Webcam non détectée** :
   - Vérifiez que la webcam n'est pas utilisée par une autre application
   - Accordez les permissions d'accès à la webcam

3. **Modèles non trouvés** :
   - Assurez-vous que les fichiers `.pt` sont dans le dossier `models/`
   - Vérifiez les chemins dans la configuration

4. **Performances lentes** :
   - Réduisez le seuil de confiance
   - Utilisez une résolution vidéo plus faible
   - Fermez les autres applications gourmandes en ressources

### Support

Pour signaler un bug ou demander de l'aide :

1. Créez une nouvelle issue avec :
   - Description détaillée du problème
   - Version de Python (`python --version`)
   - Système d'exploitation
   - Logs d'erreur complets

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Auteurs

Projet réalisé par **Nicolas Lambropoulos** dans le cadre d'un projet scolaire (PPE) à l'ECE paris.

##  Remerciements

- **YOLOv8** par Ultralytics pour la détection d'objets
- **PyTorch** pour le framework de deep learning
- **OpenCV** pour le traitement d'images
- **Supervision** pour les utilitaires de vision par ordinateur

---
