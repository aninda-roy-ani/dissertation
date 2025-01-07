import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(base_dir, img_size=224, batch_size=32):

    datagen = ImageDataGenerator(rescale=1./255)

    train_data = datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validate_data = datagen.flow_from_directory(
        os.path.join(base_dir, 'validate'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_data, validate_data, test_data