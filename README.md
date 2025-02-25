
# AI Project Report: Fruit Plant Leaf Identification and Disease Recognition

**Course:** CS251  
**Instructor:** Nazia Shahzadi  
**Team Members:**  
- Muhammad Adeel (2022331)  
- Saud Khan (2022533)  

## 1. Introduction
This project aims to develop a deep learning model for accurate classification of plant diseases from leaf images, assisting farmers in early detection and intervention.

### Objectives
- Build a robust deep learning model for plant disease classification.
- Improve early disease detection for timely treatment.
- Provide a reliable tool for agricultural disease management.
- Increase crop yield and productivity.
- Overcome limitations of manual disease detection.

## 2. Dataset Description
- **Content:** 87,000 RGB images of healthy and diseased leaves.
- **Classes:** 38 plant species affected by different diseases.
- **Augmentation:** Enhances dataset size and diversity.
- **Partitioning:** 80/20 training-validation split.
- **Test Set:** 33 images for evaluation.

## 3. Neural Network Architectures
### Custom CNN Model
- Tailored architecture for precision in plant disease classification.

### VGG16
- Pre-trained on ImageNet, extracts robust features, and ensures better performance.

### ResNet34
- Pre-trained deep CNN model with shortcut connections, ensuring stable and faster training.

### Comparative Analysis
- Evaluates performance of Custom CNN, VGG16, and ResNet34 for optimal classification accuracy.

## 4. Hybrid/Ensemble Approach
- **Ensemble Learning:** Combines Custom CNN, VGG16, and ResNet34.
- **Collective Intelligence:** Aggregates model outputs for reliable predictions.
- **Performance Boost:** Reduces individual model biases, improving classification accuracy.

## 5. Model Implementation
- **Preprocessing:** Image resizing (128x128), transformation to PyTorch tensors.
- **Training:** Model training with hyperparameter tuning.
- **Evaluation:** Metrics computation and visualization.
- **Predictions:** Tested on validation images with correct label identification.

## 6. Training Procedure and Hyperparameters
- **Optimization:** Adam optimizer (learning rate: 0.001) for stable convergence.
- **Fine-Tuning:** Batch size, optimizer selection, and learning rate adjustments.
- **Efficiency:** Ensures robust model updates, minimizing loss and enhancing accuracy.

## 7. Evaluation Results
- **Accuracy:** Increases with epochs, indicating learning progress.
- **Loss Trends:** Training and validation loss decrease over epochs.
- **Precision, Recall, F1-Score:** Evaluates classification effectiveness.
- **AUC-ROC Score:** Measures model performance across class imbalances.
- **Confusion Matrix:** Highlights correct and incorrect classifications.

## 8. Conclusion
This model enhances plant disease detection, promoting sustainable agriculture. Future improvements include increased robustness, scalability, and adaptability for real-world applications.
