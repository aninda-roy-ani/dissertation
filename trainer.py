import matplotlib.pyplot as plt


def train_model(model, train_data, validate_data, epochs=10):
    """
    Train the model and return the training history.
    """
    history = model.fit(
        train_data,
        validation_data=validate_data,
        epochs=epochs,
        verbose=1
    )
    return history


def evaluate_model(model, test_data):
    """
    Evaluate the model on the test data.
    """
    loss, accuracy = model.evaluate(test_data, verbose=1)
    return loss, accuracy


def plot_history(history, model_name, output_dir='results'):
    """
    Plot training history and save the plots to the output directory.
    """
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name.lower()}_history.png")
    plt.show()
