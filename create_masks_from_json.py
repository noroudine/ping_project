import json
import cv2
import numpy as np
import os
from pathlib import Path

def create_mask_from_json(json_path, image_path, output_mask_path):
    """
    Crée un masque binaire à partir des coordonnées des bâtiments dans le fichier JSON.
    
    Args:
        json_path: Chemin vers le fichier JSON contenant les coordonnées
        image_path: Chemin vers l'image source
        output_mask_path: Chemin où sauvegarder le masque généré
    """
    # Lire l'image pour obtenir ses dimensions
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erreur: Impossible de lire l'image {image_path}")
        return
    
    height, width = img.shape[:2]
    
    # Créer un masque vide
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Lire le fichier JSON
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Parcourir les polygones (différentes structures possibles)
        if 'buildings' in data:
            # Structure 1: liste de bâtiments
            for building in data['buildings']:
                if 'coordinates' in building:
                    coords = np.array(building['coordinates'], dtype=np.int32)
                    cv2.fillPoly(mask, [coords], 255)
        elif 'shapes' in data:
            # Structure 2: liste de formes (format labelme)
            for shape in data['shapes']:
                if shape['shape_type'] == 'polygon':
                    coords = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [coords], 255)
        elif isinstance(data, list):
            # Structure 3: liste directe de polygones
            for polygon in data:
                if 'points' in polygon:
                    coords = np.array(polygon['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [coords], 255)
        
        # Sauvegarder le masque
        cv2.imwrite(output_mask_path, mask)
        print(f"Masque créé avec succès: {output_mask_path}")
        
    except Exception as e:
        print(f"Erreur lors du traitement du fichier JSON {json_path}: {str(e)}")

def process_all_data(images_dir, output_masks_dir):
    """
    Traite tous les fichiers images et crée les masques correspondants.
    
    Args:
        images_dir: Dossier contenant les images sources
        output_masks_dir: Dossier où sauvegarder les masques générés
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_masks_dir, exist_ok=True)
    
    # Extensions d'images supportées
    image_extensions = {'.jpg', '.jpeg', '.png', '.avif'}
    
    # Parcourir tous les fichiers dans le dossier d'images
    for image_path in Path(images_dir).glob('*'):
        if image_path.suffix.lower() in image_extensions:
            # Chercher le fichier JSON correspondant
            json_path = image_path.with_suffix('.json')
            
            # Vérifier si le fichier JSON existe
            if json_path.exists():
                # Construire le chemin du masque de sortie
                mask_path = Path(output_masks_dir) / f"{image_path.stem}_mask.png"
                
                # Créer le masque
                create_mask_from_json(str(json_path), str(image_path), str(mask_path))
            else:
                print(f"Fichier JSON non trouvé pour {image_path}")

if __name__ == '__main__':
    # Définir les chemins des dossiers
    images_dir = 'Plan_De_MasseIMG'
    output_masks_dir = 'masks'
    
    process_all_data(images_dir, output_masks_dir)
