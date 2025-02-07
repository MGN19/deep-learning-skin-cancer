# Skin Cancer Classification using Deep Learning

## Project Overview
This project focuses on developing a deep-learning model for skin cancer classification using the HAM10000 dataset. The dataset contains images of skin lesions categorized into seven types of skin cancer. The goal is to build a model that can accurately classify unseen images while handling data imbalances and computational constraints.

## Dataset
The dataset consists of 10,015 labeled images along with metadata, including:
- **Age**
- **Gender**
- **Location of the lesion**
- **Diagnosis (dx)** (Target variable with seven classes):
  - Actinic keratoses and intraepithelial carcinoma / Bowen's disease (**akiec**)
  - Basal cell carcinoma (**bcc**)
  - Benign keratosis-like lesions (**bkl**)
  - Dermatofibroma (**df**)
  - Melanoma (**mel**)
  - Melanocytic nevi (**nv**)
  - Vascular lesions (**vasc**)

## Methodology
### Data Exploration
- Analyzed metadata to identify missing values and class distribution.
- Noted dataset imbalance, requiring the use of **f1-score** as a key evaluation metric.
- Decided to use only images for training, excluding metadata.

### Image Preprocessing
- Resized images from **600x450** to **150x112** to optimize computational efficiency.
- Applied label encoding to the target variable.
- Normalized pixel values to a range between **0 and 1**.
- Converted images to grayscale and tested **hair removal techniques**.
- Applied sharpening and histogram equalization for contrast enhancement.
- Due to suboptimal results, preprocessing was ultimately **not used** in the final model.

### Model Development
- Implemented a **Convolutional Neural Network (CNN)** using TensorFlow/Keras.
- Used **ImageDataGenerator** for data augmentation (rotation, flipping, shifting), but found no improvement in performance.
- Employed **grid search** to optimize hyperparameters using **Keras Tuner - Hyperband**.

### Best CNN Model Structure
- **3 Convolutional layers** (20, 60, 80 filters) with ReLU activation and max pooling.
- Flattened the output and added **3 dense layers**:
  - **352 neurons, ReLU, Dropout (30%)**
  - **256 neurons, ReLU, Dropout (10%)**
  - **32 neurons, ReLU, Dropout (30%)**
- **Final softmax layer** with 7 neurons for multi-class classification.
- Optimized using **Adam optimizer** and **sparse_categorical_crossentropy loss**.

### Model Evaluation
- Used **stratified k-fold cross-validation** to handle class imbalance.
- Achieved a weighted **f1-score of 0.74** and an accuracy of **75.42%**.
- **Confusion matrix analysis** showed that the model performed well on common classes (e.g., 'nv') but struggled with rare ones (e.g., 'df').

## Results & Conclusions
- The final model achieved an f1-score of **0.74**.
- The biggest challenge was **RAM limitations**, which forced resizing of images.
- Future improvements could include **better class balancing techniques** and **transfer learning** with pre-trained models.

## Installation & Usage
### Prerequisites
- Python 3.x
- TensorFlow/Keras
- NumPy, Pandas, Matplotlib, Seaborn, OpenCV
- Sklearn (for label encoding and k-fold validation)

### Running the Model
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/skin-cancer-classification.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the notebooks in order:
   - `01_data_exploration.ipynb`
   - `02_basic_model.ipynb`
   - `03_image_augmentation.ipynb`
   - `04_image_preprocessing.ipynb`
   - `05_grid_search.ipynb`
   - `06_model_evaluation.ipynb`

## References
[1] Agyenta & Akanzawon, "Skin Lesion Classification Based on Convolutional Neural Network".

## License
This project is open-source and available under the MIT License.

