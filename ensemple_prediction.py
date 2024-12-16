from tensorflow.keras.models import load_model
from data_loader import load_data
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data (validation and testing)
BASE_DIR = 'output_data_ham'
IMG_SIZE = 224
BATCH_SIZE = 32

_, _, test_data = load_data(BASE_DIR, IMG_SIZE, BATCH_SIZE)

# Load the trained models
mobilenet_model = load_model('models/mobilenet_skin_disease_ham.h5')
resnet_model = load_model('models/resnet_skin_disease_ham.h5')

# Evaluate MobileNet on test data
mobilenet_loss, mobilenet_accuracy = mobilenet_model.evaluate(test_data)
print(f"MobileNet Test Accuracy: {mobilenet_accuracy * 100:.2f}%")

# Reset test data generator
test_data.reset()

# Evaluate ResNet on test data
resnet_loss, resnet_accuracy = resnet_model.evaluate(test_data)
print(f"ResNet Test Accuracy: {resnet_accuracy * 100:.2f}%")

# Reset test data generator
test_data.reset()

# Predict using MobileNet
mobilenet_preds = mobilenet_model.predict(test_data, verbose=1)

# Reset test data generator
test_data.reset()

# Predict using ResNet
resnet_preds = resnet_model.predict(test_data, verbose=1)

# Ensemble Predictions (weighted or equal)
# You can adjust weights if one model performs better (e.g., 0.6 and 0.4)
ensemble_preds = (mobilenet_preds + resnet_preds) / 2

# Convert ensemble predictions to class indices
ensemble_class_preds = np.argmax(ensemble_preds, axis=1)  # Predicted class indices
true_labels = test_data.classes  # True class indices

# Calculate Ensemble Accuracy
ensemble_accuracy = accuracy_score(true_labels, ensemble_class_preds)
print(f"Ensemble Test Accuracy: {ensemble_accuracy * 100:.2f}%")

# Confusion Matrix for Ensemble
cm = confusion_matrix(true_labels, ensemble_class_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.class_indices.keys(), yticklabels=test_data.class_indices.keys())
plt.title("Confusion Matrix - Ensemble Model")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

