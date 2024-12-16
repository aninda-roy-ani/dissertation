from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


def load_data(base_dir, img_size=224, batch_size=32, validation_split=0.2):
    """
    Load ISIC dataset and split the training data into train and validation sets.
    """
    # Use ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split  # Enable validation split
    )

    # Load training and validation datasets from the same training directory
    train_data = datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),  # Assuming the ISIC dataset has a 'train' folder
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',  # Training subset
        shuffle=True  # Shuffle training data
    )

    validate_data = datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),  # Same folder as train
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',  # Validation subset
        shuffle=True  # Shuffle validation data
    )

    # Load test data without augmentation
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_data = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),  # Assuming the ISIC dataset has a 'test' folder
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Do not shuffle test data
    )

    return train_data, validate_data, test_data
