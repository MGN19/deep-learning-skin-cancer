# Skin Cancer Classification using Neural Networks

## 📌 Project Overview
This project focused on developing a deep-learning model for skin cancer classification using the HAM10000 dataset. The dataset contains images of skin lesions categorized into seven types of skin cancer. The goal was to build a model that could classify unseen images while handling data imbalances and computational constraints.

This was a **group project** developed as part of the **Deep Learning course**.

<br>

## 📂 Dataset Overview
The dataset consists of 10 015 labeled images, including:
- **Age**
- **Sex**
- **Location of the lesion**
- **Diagnosis (dx)** (Target variable with seven classes):
  - Actinic keratoses and intraepithelial carcinoma / Bowen's disease (**akiec**)
  - Basal cell carcinoma (**bcc**)
  - Benign keratosis-like lesions (**bkl**)
  - Dermatofibroma (**df**)
  - Melanoma (**mel**)
  - Melanocytic nevi (**nv**)
  - Vascular lesions (**vasc**)

<br>

## 🔬 Methodology
### 1️⃣ Data Exploration
- Analyzed metadata to identify missing values and class distribution.
- Noted dataset imbalance, requiring the use of **f1-score** as evaluation metric.
- Decided to use only images for training, excluding metadata.

<br>

### 2️⃣ Image Preprocessing
- Resized images from **600x450** to **150x112** to optimize computational efficiency.
- Applied label encoding to the target variable.
- Normalized pixel values to a range between **0 and 1**.
- Converted images to grayscale and tested hair removal techniques.
- Applied sharpening and histogram equalization for contrast enhancement.
- Due to suboptimal results, preprocessing was ultimately **not used** in the final model.

<br>

### 3️⃣ Model Development
- Implemented a **Convolutional Neural Network (CNN)** using TensorFlow/Keras.
- Used **ImageDataGenerator** for data augmentation (rotation, flipping, shifting), but found no improvement in performance.
- Employed **grid search** to optimize hyperparameters using **Keras Tuner - Hyperband**.

<br>

### 4️⃣ Model Evaluation
- Used **stratified k-fold cross-validation** to handle class imbalance.
- Achieved a weighted **f1-score of 0.74**.
- **Confusion matrix analysis** showed that the model performed well on majority classes (e.g., 'nv') but struggled with rare ones (e.g., 'df').

### 🔥 Best CNN Model Architecture
- **3 Convolutional layers** (20, 60, 80 filters) with ReLU activation and max pooling.
- Flattened the output and added **3 dense layers**:
  - **352 neurons, ReLU, Dropout (30%)**
  - **256 neurons, ReLU, Dropout (10%)**
  - **32 neurons, ReLU, Dropout (30%)**
- **Final softmax layer** with 7 neurons for multi-class classification.
- Optimized using **Adam optimizer** and **sparse_categorical_crossentropy loss**.

<br>

## 🎯 Results & Challenges  

✅ Key Achievements  

The model achieved an F1-score of 0.74.  

Successfully optimized a deep-learning model despite limited resources.  

<br>

❌ Challenges Faced  

Computational Limitations: Required image resizing due to RAM constraints.  

Class Imbalance: Rare categories impacted model accuracy.  

Preprocessing Trade-offs: Despite testing, preprocessing did not significantly improve results.  

<br>

## ⚙️ Installation & Usage
### 🔧 Prerequisites
- Python 3.x
- TensorFlow/Keras
- NumPy, Pandas, Matplotlib, Seaborn, OpenCV
- Sklearn (for label encoding and k-fold validation)

<br>

### 🚀 Running the Model
1. Clone this repository:
   ```sh
   git clone https://github.com/MGN19/deep-learning-skin-cancer.git
   ```
2. Run the notebooks in order:
   - `1_Explore.ipynb`
   - `2_ImagePreProcessing.ipynb`
   - `3_Model.ipynb`
   - `4_ImgGen.ipynb`
   - `5_GridSearch.ipynb`
  


