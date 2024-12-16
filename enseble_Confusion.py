from tensorflow.keras.models import load_model
from data_loader import load_data
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data (test data generator)
BASE_DIR = 'output_data_ham'
IMG_SIZE = 224
BATCH_SIZE = 1  # Ensure batch_size=1 for alignment

_, _, test_data = load_data(BASE_DIR, IMG_SIZE, BATCH_SIZE)

# Load pre-trained models
mobilenet_model = load_model('models/mobilenet_skin_disease.h5')
resnet_model = load_model('models/resnet_skin_disease.h5')

# Compile models for evaluation
mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate individual models
mobilenet_loss, mobilenet_accuracy = mobilenet_model.evaluate(test_data, verbose=1)
print(f"MobileNet Test Accuracy: {mobilenet_accuracy * 100:.2f}%")
test_data.reset()  # Reset generator

resnet_loss, resnet_accuracy = resnet_model.evaluate(test_data, verbose=1)
print(f"ResNet Test Accuracy: {resnet_accuracy * 100:.2f}%")
test_data.reset()  # Reset generator

# Get predictions from both models
mobilenet_preds = mobilenet_model.predict(test_data, verbose=1)
test_data.reset()  # Reset generator

resnet_preds = resnet_model.predict(test_data, verbose=1)
test_data.reset()  # Reset generator

# Weighted ensemble (adjust weights based on individual performance)
ensemble_preds = (0.6 * mobilenet_preds + 0.4 * resnet_preds) / (0.6 + 0.4)
ensemble_class_preds = np.argmax(ensemble_preds, axis=1)  # Convert probabilities to class indices
true_labels = test_data.classes  # True class indices

# Calculate ensemble accuracy
ensemble_accuracy = accuracy_score(true_labels, ensemble_class_preds)
print(f"Ensemble Test Accuracy: {ensemble_accuracy * 100:.2f}%")

# Print predictions and labels for debugging
for i in range(5):  # Print first 5 samples
    print(f"Sample {i}:")
    print(f"MobileNet Predicted: {np.argmax(mobilenet_preds[i])}, ResNet Predicted: {np.argmax(resnet_preds[i])}")
    print(f"Ensemble Predicted: {ensemble_class_preds[i]}, True Label: {true_labels[i]}")

# Confusion matrix
cm = confusion_matrix(true_labels, ensemble_class_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.class_indices.keys(), yticklabels=test_data.class_indices.keys())
plt.title("Confusion Matrix - Weighted Ensemble")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
