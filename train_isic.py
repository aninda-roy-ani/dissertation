import os
from load_data_isic import load_data
from model_builder import build_mobilenet_model, build_resnet_model
from trainer import train_model, evaluate_model, plot_history

# Directories and settings
BASE_DIR = 'output_data_isic'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 9
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create directories for results
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load data
train_data, validate_data, test_data = load_data(BASE_DIR, IMG_SIZE, BATCH_SIZE)

# Build MobileNet and ResNet models
mobilenet_model = build_mobilenet_model(INPUT_SHAPE, NUM_CLASSES)
resnet_model = build_resnet_model(INPUT_SHAPE, NUM_CLASSES)

# Compile the models
mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train MobileNet
print("Training MobileNet...")
mobilenet_history = train_model(mobilenet_model, train_data, validate_data, epochs=EPOCHS)
mobilenet_model.save('models/mobilenet_skin_disease.h5')

# Train ResNet
print("Training ResNet...")
resnet_history = train_model(resnet_model, train_data, validate_data, epochs=EPOCHS)
resnet_model.save('models/resnet_skin_disease.h5')

# Evaluate models
mobilenet_loss, mobilenet_accuracy = evaluate_model(mobilenet_model, test_data)
resnet_loss, resnet_accuracy = evaluate_model(resnet_model, test_data)

print(f"MobileNet Test Accuracy: {mobilenet_accuracy * 100:.2f}%")
print(f"ResNet Test Accuracy: {resnet_accuracy * 100:.2f}%")

# Plot histories
plot_history(mobilenet_history, "MobileNet")
plot_history(resnet_history, "ResNet")
