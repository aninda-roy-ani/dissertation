import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(base_dir, img_size=224, batch_size=32):
    """
    Load data from a directory with train, validate, and test splits.
    """
    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

    train_generator = datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validate_generator = datagen.flow_from_directory(
        os.path.join(base_dir, 'validate'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validate_generator, test_generator
