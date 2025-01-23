import cv2
import pytesseract
from PIL import Image
import os

def extract_rotate_and_resize_plan_from_images(input_folder, output_folder, new_size, search_text, angle):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)

            if search_text.lower() in text.lower():
                # Get the image dimensions
                (h, w) = img.shape[:2]
                # Calculate the center of the image
                center = (w // 2, h // 2)
                # Perform the rotation
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h))
                # Resize the rotated image
                img_resized = cv2.resize(rotated, new_size)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, img_resized)

input_folder = 'Plan'
output_folder = 'Plan/output'
new_size = (800, 600)  # Replace with desired dimensions
search_text = "plan de masse"
angle = 90  # Replace with desired rotation angle

extract_rotate_and_resize_plan_from_images(input_folder, output_folder, new_size, search_text, angle)