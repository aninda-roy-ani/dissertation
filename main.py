import os
from trainer import train_model


def main():

    # Set dataset directory and model parameters
    base_dir = 'output_data_ham'  # Update this path to your dataset
    img_size = 224
    batch_size = 32
    epochs = 15

    # Create necessary directories
    os.makedirs('models', exist_ok=True)

    # Train MobileNet
    print("Training MobileNet...")
    train_model(base_dir, model_name='mobilenet', epochs=epochs, batch_size=batch_size, img_size=img_size)

    # Train ResNet
    print("Training ResNet...")
    train_model(base_dir, model_name='resnet', epochs=epochs, batch_size=batch_size, img_size=img_size)


if __name__ == "__main__":
    main()
