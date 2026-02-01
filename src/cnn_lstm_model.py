import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_lstm(num_classes=10):
    model = models.Sequential()

    # Input: (time_steps, height, width, channels)
    model.add(layers.Input(shape=(10, 128, 128, 3)))

    model.add(layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), activation='relu')
    ))
    model.add(layers.TimeDistributed(
        layers.MaxPooling2D((2, 2))
    ))

    model.add(layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), activation='relu')
    ))
    model.add(layers.TimeDistributed(
        layers.MaxPooling2D((2, 2))
    ))

    model.add(layers.TimeDistributed(layers.Flatten()))

    model.add(layers.LSTM(128))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = build_cnn_lstm()
    model.summary()
