from tensorflow.keras.models import load_model
from data_loader import load_data

# Load data (only needed for validation and testing)
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

# Evaluate ResNet on test data
resnet_loss, resnet_accuracy = resnet_model.evaluate(test_data)
print(f"ResNet Test Accuracy: {resnet_accuracy * 100:.2f}%")
