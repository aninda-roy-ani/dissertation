import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the test dataset
test_data_dir = "output_data_ham/test"  # Path to the test data directory
img_size = (224, 224)
batch_size = 32

# Set up the data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# Load the saved models
mobilenet_model = tf.keras.models.load_model("models/mobilenet_best.keras")
resnet_model = tf.keras.models.load_model("models/resnet_best.keras")

# Get true labels and class indices
true_labels = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# Generate predictions for MobileNet
mobilenet_predictions = mobilenet_model.predict(test_generator)
mobilenet_pred_classes = np.argmax(mobilenet_predictions, axis=1)

# Generate predictions for ResNet
resnet_predictions = resnet_model.predict(test_generator)
resnet_pred_classes = np.argmax(resnet_predictions, axis=1)

# Compute confusion matrices
mobilenet_cm = confusion_matrix(true_labels, mobilenet_pred_classes)
resnet_cm = confusion_matrix(true_labels, resnet_pred_classes)

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, class_names, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(mobilenet_cm, class_names, "Confusion Matrix for MobileNet")
plot_confusion_matrix(resnet_cm, class_names, "Confusion Matrix for ResNet")

# Classification reports
print("Classification Report for MobileNet:")
print(classification_report(true_labels, mobilenet_pred_classes, target_names=class_names))

print("Classification Report for ResNet:")
print(classification_report(true_labels, resnet_pred_classes, target_names=class_names))
