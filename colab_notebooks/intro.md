## Brief Intro

The `end-to-end-dog-vision` notebook contains code for building, training, and testing a model built upon MobileNetV2. No data augmentation or fine tuning is performed.

The`end-to-end-dog-vision-data-augmented` notebook includes data augmentation in the preprocessing pipeline. There are significant performance  improvements indicating less overfitting than the original notebook, involving an increase in accuracy of the full model on test data (confirmed via late submission to Kaggle competition) from **88.02%** (pre-augmentation) to **98.54%** (post-augmentation).

<img width="1135" height="239" alt="Screenshot 2026-01-22 at 10 13 19 PM" src="https://github.com/user-attachments/assets/0d7053ec-39bd-4726-b9e0-663250b29264" />

Future notebooks added to this folder will implement next steps, including but not limited to fine tuning and dropout, along with trying different models and evaluation metrics.
