import numpy as np
from keras.src.callbacks import EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from layers import models

from sklearn.model_selection import train_test_split
from tensorflow.python.layers import layers



X=(data['images'])
y=(data['labels'])

unique_shapes = set([image.shape for image in X])
print(unique_shapes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def build_model(filters = 32, kernel_size = 3, dense_units = 64, dropout_rate = 0.5):
    model = models.Sequential()
    model.add(layers.Conv2D(filters, (kernel_size, kernel_size), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 3)))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu'))

    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())

    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(8, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

train_datagen = ImageDataGenerator(rescale = 1. / 255)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
test_generator = test_datagen.flow(X_test, y_test, batch_size=32)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

model = build_model(filters=32, kernel_size=3, dense_units=64, dropout_rate=0.5)
history = model.fit(train_generator, validation_data=test_generator, epochs=100, callbacks=[early_stopping])

loss, accuracy = model.evaluate(test_generator)
print('Test accuracy:', accuracy)

