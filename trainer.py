import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model_builder import build_model
from data_loader import load_data

def train_model(base_dir, model_name, epochs=10, batch_size=32, img_size=224):

    # Load data
    train_data, validate_data, test_data = load_data(base_dir, img_size, batch_size)

    # Build model
    model = build_model(model_name, input_shape=(img_size, img_size, 3), num_classes=train_data.num_classes)

    # Callbacks
    checkpoint = ModelCheckpoint(f"models/{model_name}_best.keras", monitor='val_accuracy', save_best_only=True, mode='max')

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    # Train model
    history = model.fit(
        train_data,
        validation_data=validate_data,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping]
    )

    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    return history