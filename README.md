# English Digit Recognition

This project builds a handwritten digit recognizer (0-9) from screen-captured drawings. The implementation is notebook-first and uses a classical machine learning pipeline based on a linear Support Vector Classifier (SVC).

## Project Scope

The main workflow is implemented in `digit_recognition.ipynb`:

1. Capture digit images from a fixed screen region.
2. Build a CSV dataset from images.
3. Train and evaluate a linear SVC model.
4. Predict a newly drawn digit from a live capture.

The file `main.py` is currently a placeholder and is not the active application entry point.

## Technical Workflow

### 1. Image Capture

- Captures are taken with `pyscreenshot` from a fixed bounding box: `(80, 250, 800, 800)`.
- Images are organized by class label folders (`0` to `9`).

### 2. Dataset Generation

- Each image is converted to grayscale, blurred, and resized to `64x64`.
- Pixel values are thresholded to binary values (`0` or `1`).
- Each sample is stored in `digit_recognition.csv` as:
	- `label`
	- `pixel0` ... `pixel4095`

### 3. Model Training

- Data is shuffled and split (`test_size=0.2`).
- Model: `sklearn.svm.SVC(kernel='linear', random_state=6)`.
- Accuracy is computed with `sklearn.metrics.accuracy_score`.
- Model artifact is saved under `model/`.

### 4. Live Prediction

- Captures a new drawing from the same screen region.
- Applies the same preprocessing and feature extraction.
- Loads the trained model and predicts the digit.

## Repository Structure

- `digit_recognition.ipynb`: full capture/train/predict pipeline and GUI cells.
- `digit_recognition.csv`: generated dataset (label + 4096 binary features).
- `model/`: trained model artifact (`digit_recognition_model.pkl`).
- `image/`: class-wise training images (`0` to `9`).
- `captured_images/`: additional captured images (`0` to `9`).
- `image2/`: auxiliary image set (`0` to `9`).
- `img/`: temporary prediction capture (for example `img.png`).
- `main.py`: placeholder script.

## Requirements

- Python 3.8+
- Windows environment (current notebook paths and Paint launch are Windows-specific)
- Packages:
	- `numpy`
	- `pandas`
	- `opencv-python`
	- `scikit-learn`
	- `joblib`
	- `Pillow`
	- `pyscreenshot`
	- `matplotlib` (used in notebook visualization)

Install dependencies:

```bash
pip install numpy pandas opencv-python scikit-learn joblib Pillow pyscreenshot matplotlib
```

## How To Run

1. Open `digit_recognition.ipynb`.
2. Run cells in order from top to bottom.
3. For training from scratch:

  - ensure images are present under `image/0` ... `image/9`
  - regenerate `digit_recognition.csv`
  - train and save the model

4. Run the prediction section to classify a new drawing.

## Important Notes

- Hardcoded paths exist in GUI cells (for example Paint path and capture output path). Update these paths to match your machine.
- Capture coordinates are fixed; if your display layout is different, update the bounding box.
- CSV writing uses append mode (`'a'`). Re-running dataset generation without cleanup will duplicate headers and rows.
- Two model names are used in the notebook (`model/digit_recognition` and `model/digit_recognition_model.pkl`). Keep one consistent model path for reliable inference.
