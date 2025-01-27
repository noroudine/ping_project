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
├── 📄 train_segmentation.py  (Pour entraîner le sy