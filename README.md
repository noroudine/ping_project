# Image Processing Project

This project extracts, rotates, and resizes images containing specific text using OpenCV and Tesseract OCR.

## Prerequisites

- Python 3.x
- OpenCV
- Tesseract OCR
- Pillow

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python pytesseract pillow
    ```

3. Install Tesseract OCR:
    - **macOS**: 
        ```sh
        brew install tesseract
        ```
    - **Windows**: Download the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and follow the installation instructions.

## Usage

1. Place the images you want to process in the `Plan` folder.

2. Run the script:
    ```sh
    python generate_dataset.py
    ```

3. The processed images will be saved in the `Plan/output` folder.

## Configuration

You can configure the script by modifying the following variables in `generate_dataset.py`:

- `input_folder`: The folder containing the input images.
- `output_folder`: The folder where the processed images will be saved.
- `new_size`: The new size for the resized images (e.g., `(800, 600)`).
- `search_text`: The text to search for in the images (e.g., `"plan de masse"`).
- `angle`: The angle by which to rotate the images (e.g., `90`).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
