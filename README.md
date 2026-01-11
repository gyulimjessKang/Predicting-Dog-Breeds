# End-to-End Dog Breed Identification  
An end-to-end, multi-class image classification pipeline that predicts a dog’s breed from a photo using transfer learning (MobileNetV2 via TensorFlow Hub). Built and run in a Google Colab notebook, with a data → tensors → batches → train → evaluate → predict → export workflow.

---

## What this project does
- Loads a labeled dog image dataset (`labels.csv` + `train/` images)
- Builds a **TensorFlow `tf.data` input pipeline** (preprocess + batching)
- Trains a **transfer learning model** (MobileNetV2 feature extractor with softmax head)
- Tracks training with **TensorBoard** and stabilizes training with **EarlyStopping**
- Generates predictions for:
  - a validation split (to evaluate performance)
  - a test set (to create a Kaggle-style submission file)
  - custom images (for quick sanity checks)
Note: some files (e.g. train and test folders) were too large to include in this repo.
---

## Notebook workflow (step-by-step)

### 1) Define the problem
Multi-class classification: given a dog image, predict one of the many possible breeds.

### 2) Load and explore the data
- Read labels from `labels.csv` (breed names per image id)
- Inspect class distribution (bar plot of breed counts)
- Construct full image filepaths from image ids
- Run sanity checks:
  - number of labels matches number of filenames
  - number of filenames matches the number of files in the training directory
- Preview sample images to confirm paths and labels look correct

### 3) Encode labels
- Extract `unique_breeds`
- Convert breed strings into **one-hot encoded** boolean arrays (`boolean_labels`)
  - This matches the model’s categorical softmax output

### 4) Create a validation split
- Split a subset of the dataset into train/validation using `train_test_split`
- Uses `test_size=0.2` for validation and a fixed `random_state` for reproducibility
- A notebook parameter controls how many images are used during experimentation:
  - `NUM_IMAGES` (slider) for faster iteration before full training

### 5) Preprocess images into tensors
A preprocessing function turns image paths into tensors ready to be fed as inputs into the model.

### 6) Build input pipelines with `tf.data`
A batching helper builds datasets for different phases:
- **Training**: shuffles filepaths/labels, maps `(image, label)` preprocessing, batches
- **Validation**: no shuffling, maps preprocessing, batches
- **Test**: filepaths only (no labels), maps preprocessing, batches

This keeps training efficient and scalable, and makes it easy to swap between modes.

### 7) Build the model (transfer learning)
Model architecture:
- **Base**: MobileNetV2 from TensorFlow Hub (feature extractor)
- **Head**: Dense layer with `softmax` over the number of breeds

Compiled with:
- Loss: `CategoricalCrossentropy`
- Optimizer: `Adam`
- Metric: `accuracy`

### 8) Add training callbacks
- **TensorBoard**: logs metrics for each experiment run
- **EarlyStopping**: monitors `val_accuracy` with a patience of 3 epochs, helps mitigate overfitting

### 9) Train a model on a subset (fast iteration)
- Train using the training batches and validate on validation batches
- Use TensorBoard logs to inspect learning curves

### 10) Evaluate + visualize predictions
- Generate prediction probabilities on validation batches
- Convert probabilities → predicted label via `argmax`
- Plot predictions and confidence distributions:
  - predicted label vs true label
  - prediction probability breakdown

### 11) Save and reload trained models
Utility functions to save and load models enables:
- resuming evaluation without retraining, saving time
- comparing multiple experiments 

### 12) Train on the full dataset
Once the pipeline works on a subset:
- Create batches for the full training set
- Train a “full model” using the same architecture + callbacks

### 13) Predict on the test set and format submission
- Create test batches (filepaths only)
- Generate predictions for each test image
- Format a submission dataframe:
  - `id` column + one column per breed
  - fill each breed column with predicted probabilities
- Export predictions to CSV

### 14) Predict on custom images
- Load external/custom images
- Preprocess them with the same function
- Run inference and visualize predicted breed(s)

---

## Tech stack
- Python
- TensorFlow
- TensorFlow Hub (transfer learning)
- NumPy / Pandas
- Scikit-learn (train/validation split)
- Matplotlib (visuals)
- TensorBoard

---

## How to run (recommended: Colab)
1. Open the notebook in Google Colab.
2. Ensure your dataset paths match your environment. Data can be found here: https://www.kaggle.com/competitions/dog-breed-identification. The notebook expects dataset files in a Google Drive directory like:
   - `.../dog_breed_identification/labels.csv`
   - `.../dog_breed_identification/train/`
   - `.../dog_breed_identification/test/`
3. Run the notebook top-to-bottom:
   - start with a smaller `NUM_IMAGES` to iterate quickly
   - then train the full model once everything checks out

---

## Outputs
- Saved model files (`.h5`) for both:
  - a “subset-trained” experiment model
  - a full-dataset trained model
- TensorBoard logs for experiment tracking
- Cleanly formatted CSV of test set predictions
- Example predictions on custom images

---

## Next steps (planned improvements)

### 1) Data augmentation (in progress)
Add augmentation during training (e.g., flips/rotations/zoom/color jitter) and measure impact on:
- validation accuracy
- generalization on custom images
- training stability / overfitting

### 2) Fine-tuning the base model
After training the top classifier head:
- unfreeze upper MobileNetV2 layers
- train with a lower learning rate

### 3) Experiment tracking + comparison table
Turn experiments into a repeatable “run sheet”:
- dataset size used
- augmentation on/off
- learning rate, batch size, epochs
- best val accuracy
- link to model artifact + TensorBoard run

### 4) Confusion matrix + per-class performance
Accuracy is nice, but:
- which breeds are most confused?
- are there rare classes that underperform?
Add:
- confusion matrix
- per-class precision/recall (or top-k accuracy)

---

## Repository notes
- The core logic lives in the notebook:
  - preprocessing (`process_image`)
  - batching (`create_data_batches`)
  - model creation (`create_model`)
  - training (`train_model`)
  - save/load utilities
