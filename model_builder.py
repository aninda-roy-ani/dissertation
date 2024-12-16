from tensorflow.keras.applications import MobileNet, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

def build_mobilenet_model(input_shape, num_classes):

    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),  # Add Dropout
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # Add L2 Regularization
        Dropout(0.5),  # Add another Dropout
        Dense(num_classes, activation='softmax')
    ])
    return model


def build_resnet_model(input_shape, num_classes):
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),  # Add Dropout
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # Add L2 Regularization
        Dropout(0.5),  # Add another Dropout
        Dense(num_classes, activation='softmax')
    ])
    return model
