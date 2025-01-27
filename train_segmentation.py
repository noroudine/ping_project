import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from albumentations import Compose, HorizontalFlip, RandomRotate90, RandomBrightnessContrast, ShiftScaleRotate, ElasticTransform, GridDistortion, OpticalDistortion, OneOf

def dice_coef(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def augment_data(image, mask, num_augmentations=5):
    """
    Augmente les données en appliquant des transformations aléatoires.
    Retourne les images et masques augmentés.
    """
    # Configuration d'augmentation plus agressive
    transform = Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.7
        ),
        OneOf([
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            GridDistortion(p=0.5),
            OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        RandomBrightnessContrast(p=0.5),
    ])
    
    augmented_images = []
    augmented_masks = []
    
    for _ in range(num_augmentations):
        augmented = transform(image=image, mask=mask)
        augmented_images.append(augmented['image'])
        augmented_masks.append(augmented['mask'])
    
    return augmented_images, augmented_masks

def create_unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Normalisation initiale plus forte
    x = Lambda(lambda x: (x / 127.5) - 1)(inputs)
    
    # Encoder avec plus de dropout
    conv1 = Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(256, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge avec plus de filtres
    conv4 = Conv2D(512, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.4)(conv4)
    conv4 = Conv2D(512, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    
    # Decoder avec skip connections améliorées
    up5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4)
    merge5 = concatenate([conv3, up5])
    conv5 = Conv2D(256, 3, activation=None, padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(256, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = concatenate([conv2, up6])
    conv6 = Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    
    up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv1, up7])
    conv7 = Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    
    # Couche de sortie avec activation sigmoid
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

class SaveTrainingPlotsCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_freq=1):
        super(SaveTrainingPlotsCallback, self).__init__()
        self.save_freq = save_freq
        self.history = {'loss': [], 'val_loss': [], 'dice_coef': [], 'val_dice_coef': []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Mettre à jour l'historique
        for metric in self.history.keys():
            if metric in logs:
                self.history[metric].append(logs[metric])
        
        if (epoch + 1) % self.save_freq == 0:
            plt.figure(figsize=(12, 4))
            
            # Graphique de la perte
            plt.subplot(1, 2, 1)
            if self.history['loss']:
                plt.plot(self.history['loss'], label='Training Loss')
            if self.history['val_loss']:
                plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Graphique du coefficient Dice
            plt.subplot(1, 2, 2)
            if self.history['dice_coef']:
                plt.plot(self.history['dice_coef'], label='Training Dice Coefficient')
            if self.history['val_dice_coef']:
                plt.plot(self.history['val_dice_coef'], label='Validation Dice Coefficient')
            plt.title('Model Dice Coefficient')
            plt.xlabel('Epoch')
            plt.ylabel('Dice Coefficient')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_plots.png')
            plt.close()

def load_data(image_dir, mask_dir, target_size=(256, 256)):
    images = []
    masks = []
    
    # Charger toutes les images originales (non augmentées)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpeg', '.jpg', '.png', '.avif')) and not f.startswith('aug_')]
    
    print(f"Nombre total d'images originales trouvées : {len(image_files)}")
    
    for img_file in image_files:
        try:
            # Charger l'image
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, f"{os.path.splitext(img_file)[0]}_mask.png")
            
            if os.path.exists(mask_path):
                # Charger et prétraiter l'image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Erreur lors du chargement de l'image : {img_path}")
                    continue
                
                # S'assurer que l'image a 3 canaux
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Redimensionner l'image
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32) / 255.0  # Normalisation
                
                # Charger et prétraiter le masque
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Erreur lors du chargement du masque : {mask_path}")
                    continue
                
                # Redimensionner le masque
                mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(np.float32) / 255.0  # Normalisation
                mask = (mask > 0.5).astype(np.float32)  # Binarisation
                
                # Vérifier que le masque contient bien des zones positives
                if np.sum(mask) > 0:
                    images.append(img)
                    masks.append(np.expand_dims(mask, axis=-1))
                    
                    # Augmenter les données
                    aug_images, aug_masks = augment_data(img, mask)
                    images.extend(aug_images)
                    masks.extend([np.expand_dims(m, axis=-1) for m in aug_masks])
                else:
                    print(f"Attention : Le masque {mask_path} ne contient aucune zone positive")
            else:
                print(f"Masque non trouvé pour {img_file}")
        except Exception as e:
            print(f"Erreur lors du traitement de {img_file}: {str(e)}")
            continue
    
    print(f"Nombre d'images chargées avec succès : {len(images)}")
    
    if len(images) == 0:
        raise ValueError("Aucune image n'a pu être chargée!")
    
    # Convertir en tableaux numpy avec la bonne forme
    X = np.array(images, dtype=np.float32)
    y = np.array(masks, dtype=np.float32)
    
    print(f"Forme des données : X={X.shape}, y={y.shape}")
    
    return X, y

def train_model():
    print("Chargement des données...")
    
    # Définir la taille d'entrée
    input_size = (256, 256)
    
    # Charger les données
    images, masks = load_data('Plan_De_MasseIMG', 'masks', target_size=input_size)
    
    # Split en train/validation
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    
    print(f"Nombre d'images d'entraînement : {len(X_train)}")
    print(f"Nombre d'images de validation : {len(X_val)}")
    
    # Créer le modèle
    model = create_unet(input_size=(*input_size, 3))
    
    # Learning rate schedule avec warm-up
    initial_learning_rate = 1e-5
    warmup_epochs = 5
    total_epochs = 300
    
    def scheduler(epoch):
        if epoch < warmup_epochs:
            return initial_learning_rate * ((epoch + 1) / warmup_epochs)
        else:
            # Exponential decay après le warm-up
            decay_rate = 0.95
            return initial_learning_rate * (decay_rate ** (epoch - warmup_epochs))
    
    # Compiler le modèle
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=initial_learning_rate),
        loss=bce_dice_loss,
        metrics=[dice_coef, 'binary_accuracy']
    )
    
    # Créer le dossier pour sauvegarder les modèles s'il n'existe pas
    os.makedirs('model_checkpoints', exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'model_checkpoints/model_epoch_{epoch:02d}_val_dice_{val_dice_coef:.4f}.h5',
            save_best_only=False,
            monitor='val_dice_coef',
            mode='max',
            save_freq='epoch'
        ),
        ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_dice_coef',
            mode='max',
            save_freq='epoch'
        ),
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        EarlyStopping(
            monitor='val_dice_coef',
            patience=20,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        SaveTrainingPlotsCallback(save_freq=1)
    ]
    
    # Entraîner le modèle
    print("Début de l'entraînement...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=16,
        epochs=total_epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Sauvegarder le modèle final
    model.save('final_model.h5')
    print("Modèle sauvegardé avec succès!")
    
    # Sauvegarder l'historique d'entraînement
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['dice_coef'], label='Training Dice')
    plt.plot(history.history['val_dice_coef'], label='Validation Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['binary_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
    plt.title('Binary Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == '__main__':
    train_model()
