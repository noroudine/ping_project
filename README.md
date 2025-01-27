# Projet de Détection de Bâtiments

Ce projet utilise l'intelligence artificielle pour détecter et délimiter automatiquement les bâtiments dans des images aériennes.

## Vue d'Ensemble

### Objectif du Projet
Le système peut automatiquement reconnaître et délimiter les bâtiments sur des images aériennes, comme si quelqu'un traçait leur contour avec un crayon.

### Comment ça marche ?

1. **L'Apprentissage**
   - Le système apprend à partir d'exemples d'images avec leurs bâtiments déjà délimités
   - Comme un enfant qui apprend : on lui montre des exemples jusqu'à ce qu'il puisse reconnaître les formes par lui-même

2. **Le Traitement des Images**
   - Chaque image est préparée pour être plus facile à analyser :
     - Redimensionnement à une taille standard
     - Ajustement de la luminosité
     - Création de variations (rotation, miroir) pour un meilleur apprentissage

3. **Le Modèle (U-Net)**
   - Fonctionne comme un scanner sophistiqué qui :
     - Analyse l'image en détail, niveau par niveau
     - Apprend à reconnaître les formes des bâtiments
     - Dessine leur contour

## Structure Technique du Projet

```
📁 PingProject/
├── 📄 train_segmentation.py  (Pour entraîner le système)
├── 📄 test_model.py         (Pour tester le système)
├── 📄 model.py              (Le cerveau du système)
└── 📄 create_masks_from_json.py (Pour préparer les données)
```

### Les Composants Principaux

1. **L'Entraîneur (`train_segmentation.py`)**
   - Agit comme un professeur qui :
     - Prend les images de bâtiments
     - Les montre au système
     - Vérifie si les réponses sont correctes
     - Aide le système à s'améliorer
   - Répète le processus 300 fois pour optimiser l'apprentissage

2. **L'Évaluateur (`test_model.py`)**
   - Fonctionne comme un examinateur qui :
     - Prend de nouvelles images
     - Demande au système de trouver les bâtiments
     - Note la précision des réponses
     - Calcule un score de performance

3. **Le Cerveau (`model.py`)**
   - C'est le système lui-même qui :
     - Analyse les images en détail
     - Identifie les caractéristiques importantes
     - Décide où sont les bâtiments
     - Dessine leurs contours

4. **Le Préparateur (`create_masks_from_json.py`)**
   - Agit comme un assistant qui :
     - Lit les annotations (fichiers JSON)
     - Crée des masques pour les bâtiments
     - Prépare les données pour l'entraînement

### Le Processus Complet

1. **Préparation des Données**
   - Redimensionnement des images à 256x256 pixels
   - Création des masques pour identifier les bâtiments
   - Préparation des données d'entraînement

2. **Phase d'Entraînement**
   - Présentation de nombreuses images au système
   - Tentatives de détection des bâtiments
   - Correction et amélioration progressive
   - Sauvegarde régulière des progrès

3. **Phase de Test**
   - Test sur de nouvelles images
   - Génération des contours de bâtiments
   - Vérification de la précision

4. **Stockage et Résultats**
   - Sauvegarde de l'apprentissage dans des fichiers `.h5`
   - Mesure de la performance (score entre 0 et 1)
   - Actuellement : score d'environ 0.11

## Installation et Utilisation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Entraîner le modèle :
```bash
python train_segmentation.py
```

4. Tester le modèle :
```bash
python test_model.py
```

## État Actuel et Perspectives

- Le système comprend les bases de la détection
- Performance actuelle : 11% de précision
- Améliorations prévues :
  - Augmentation du jeu de données
  - Optimisation de l'architecture
  - Ajustement des paramètres d'entraînement

## Contributions

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer des améliorations
- Partager des idées d'optimisation
