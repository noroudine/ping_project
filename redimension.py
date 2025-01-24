import cv2
import imgaug.augmenters as iaa
import os
import numpy as np

input_folder = 'Plan'
output_folder = 'Plan'

os.makedirs(output_folder, exist_ok=True)

augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),  # Miroir horizontal avec une probabilité de 50%
    iaa.Affine(rotate=(-20, 20)),  # Rotation entre -20 et 20 degrés
    iaa.GaussianBlur(sigma=(0, 1.0)),  # Flou gaussien
    iaa.Multiply((0.8, 1.2)),  # Variation de luminosité
])

for filename in os.listdir(input_folder):
    if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
        img = cv2.imread(os.path.join(input_folder, filename))
        for i in range(50):  # Générer 50 nouvelles versions par image
            augmented_img = augmenters(image=img)
            cv2.imwrite(os.path.join(output_folder, f"aug_{i}_{filename}"), augmented_img)

print("Augmentation terminée !")