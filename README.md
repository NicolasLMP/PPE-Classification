# PPE- Classification et comptage de poisson

Ce projet propose une solution complète pour la détection, la classification et le comptage automatique de poissons dans des vidéos sous-marines, à l’aide de modèles d’intelligence artificielle (YOLOv8 pour la détection, ResNet18 pour la classification) et d’une interface graphique conviviale.

## Fonctionnalités principales
- **Détection et classification** de poissons parmi 22 espèces méditerranéennes.
- **Comptage automatique** et suivi des individus pour éviter le double comptage.
- **Interface graphique** (Tkinter) permettant :
  - Lancement/arrêt de l’analyse vidéo (webcam ou fichier)
  - Visualisation en temps réel des détections et classifications
  - Affichage du comptage par espèce
  - Réglage du seuil de confiance
  - Export des résultats au format CSV
  - Génération de rapports graphiques

## Liste des espèces prises en charge
Anthias, Atherine, Bar européen, Bogue, Carangue, Daurade Royale, Daurade rose, Eperlan, Girelle, Gobie, Grande raie pastenague, Grande vive, Grondin, Maquereau, Mérou, Mostelle, Mulet cabot, Muraine, Orphie, Poisson scorpion, Rouget, Sole commune

## Installation
1. **Cloner ce dépôt**
2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
3. **Placer vos vidéos à analyser** dans le dossier du projet

## Utilisation
- Lancer l’application principale :
  ```bash
  python app.py
  ```
- Utiliser l’interface pour charger une vidéo ou la webcam, démarrer l’analyse, visualiser les résultats et générer des rapports.



## Démonstration vidéo
Une démonstration du système est disponible ici : [Lien Google Drive](https://drive.google.com/file/d/1ctbgBSiCJYhsyPaWc5_azhEsEapO8-NO/view?usp=sharing)

## Auteurs
Projet réalisé par le groupe #PPE24-T-297.


