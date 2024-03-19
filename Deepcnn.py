import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


# Define your CNN model
def get_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(CONST.IMG_SIZE, CONST.IMG_SIZE, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Model prepared...')
    return model


# Create an instance of your CNN model
model = get_model()


# Define GradCAM class using pytorch_grad_cam
class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, reshape_transform=None):
        super(GradCAM, self).__init__(model, target_layers, reshape_transform)

    def get_cam_weights(self, input_tensor, target_layer, target_category, activations, grads):
        return np.mean(grads, axis=(2, 3))


# Example usage of GradCAM
grad_cam = GradCAM(model=model, target_layers="target_layer_name")

# Now you can use grad_cam to generate CAMs for your CNN model
