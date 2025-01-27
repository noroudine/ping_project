import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from train_segmentation import dice_coef, bce_dice_loss

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    # Charger l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionner l'image
    image = cv2.resize(image, target_size)
    
    # Normaliser l'image
    image = image.astype('float32')
    
    return image

def predict_mask(model, image):
    # Préparer l'image pour la prédiction
    image_batch = np.expand_dims(image, 0)
    
    # Faire la prédiction
    predicted_mask = model.predict(image_batch)
    
    # Normaliser la prédiction
    predicted_mask = (predicted_mask > 0.5).astype(np.float32)
    predicted_mask = np.squeeze(predicted_mask)
    
    return predicted_mask

def plot_results(image, true_mask, predicted_mask, save_path):
    plt.figure(figsize=(15, 5))
    
    # Image originale
    plt.subplot(1, 3, 1)
    plt.imshow(image.astype('uint8'))
    plt.title('Image Originale')
    plt.axis('off')
    
    # Masque réel
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask, cmap='gray')
    plt.title('Masque Réel')
    plt.axis('off')
    
    # Masque prédit
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Masque Prédit')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_model():
    # Charger le meilleur modèle
    print("Chargement du modèle...")
    model_path = 'model_checkpoints/model_epoch_45_val_dice_0.1151.h5'
    model = load_model(model_path, custom_objects={
        'dice_coef': dice_coef,
        'bce_dice_loss': bce_dice_loss
    })
    
    # Créer le dossier pour les résultats
    os.makedirs('test_results', exist_ok=True)
    
    # Charger quelques images de test
    image_dir = 'Plan_De_MasseIMG'
    
    # Prendre quelques images augmentées pour le test
    test_images = [
        'aug_100_plan1.jpeg',
        'aug_102_plan5.jpeg',
        'aug_105_plan1.jpeg',
        'aug_107_plan7.jpeg',
        'aug_110_plan4.jpeg'
    ]
    
    for idx, image_file in enumerate(test_images):
        print(f"\nTraitement de l'image {image_file}...")
        
        # Charger l'image
        image_path = os.path.join(image_dir, image_file)
        if not os.path.exists(image_path):
            print(f"Image non trouvée: {image_path}")
            continue
            
        image = load_and_preprocess_image(image_path)
        
        # Charger le masque réel (fichier JSON correspondant)
        json_file = image_file.rsplit('.', 1)[0] + '.json'
        json_path = os.path.join(image_dir, json_file)
        
        if not os.path.exists(json_path):
            print(f"Fichier JSON non trouvé: {json_path}")
            continue
        
        # Créer un masque temporaire à partir du JSON
        import json
        from PIL import Image, ImageDraw
        
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Créer un masque vide
        mask = Image.new('L', (256, 256), 0)
        draw = ImageDraw.Draw(mask)
        
        # Dessiner les polygones à partir du JSON
        if isinstance(json_data, dict) and 'shapes' in json_data:
            # Format labelme
            shapes = json_data['shapes']
            for shape in shapes:
                if shape['shape_type'] == 'polygon':
                    points = shape['points']
                    # Convertir les points en coordonnées relatives
                    points = [(x * 256 / image.shape[1], y * 256 / image.shape[0]) for x, y in points]
                    draw.polygon(points, fill=255)
        elif isinstance(json_data, list):
            # Format personnalisé
            for annotation in json_data:
                if 'coordinates' in annotation:
                    points = annotation['coordinates']
                    # Convertir les points en coordonnées relatives
                    points = [(x * 256 / image.shape[1], y * 256 / image.shape[0]) for x, y in points]
                    draw.polygon(points, fill=255)
        
        # Convertir le masque en array numpy
        true_mask = np.array(mask)
        true_mask = (true_mask > 128).astype(np.float32)
        
        # Prédire le masque
        predicted_mask = predict_mask(model, image)
        
        # Sauvegarder les résultats
        save_path = os.path.join('test_results', f'result_{idx+1}.png')
        plot_results(image, true_mask, predicted_mask, save_path)
        
        # Calculer et afficher les métriques
        dice = dice_coef(true_mask, predicted_mask).numpy()
        print(f"Dice coefficient pour {image_file}: {dice:.4f}")

if __name__ == '__main__':
    test_model()
