from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

data = np.load("training_set.npz")

X=(data['images'])
y=(data['labels'])

print(X.shape)
print(y.shape)

X_train_val, X_test, y_train_val, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
labels = {
    0: 'Basophil',
    1: 'Eosinophil',
    2: 'Erythroblast',
    3: 'Immature granulocytes',
    4: 'Lymphocyte',
    5: 'Monocyte',
    6: 'Neutrophil',
    7: 'Platelet'
}

unique_labels=list(labels.values())

X_train, X_val, y_train, y_val = train_test_split(X_train_val,y_train_val,test_size=0.2,random_state=42,stratify=y_train_val)

train_rescaled=ImageDataGenerator(rescale=1./255)
val_rescaled=ImageDataGenerator(rescale=1./255)
test_rescaled=ImageDataGenerator(rescale=1./255)

train_normalized = train_rescaled.flow(X_train, y_train, batch_size=32)
val_normalized = val_rescaled.flow(X_val, y_val, batch_size=32)
test_normalized = test_rescaled.flow(X_test, y_test, batch_size=32)

input = (X_train.shape[1], X_train.shape[2], 3)

def build_model(input_shape, kernel_size, pool_size, num_filters, dense_units, dropout_rate):
    model = tfk.Sequential()

#aggiungere il primo layer convoluzionale e il suo pooling
    model.add(tfkl.Conv2D(num_filters,kernel_size, input_shape=input_shape, activation='relu'))
    model.add(tfkl.MaxPooling2D(pool_size=pool_size))

#aggiungere il secondo layer convoluzionale e il suo pooling
    model.add(tfkl.Conv2D(num_filters*2,kernel_size, activation='relu'))
    model.add(tfkl.MaxPooling2D(pool_size=pool_size))

#aggiungere il terzo layer convoluzionale e il suo pooling
    model.add(tfkl.Conv2D(num_filters*4,kernel_size, activation='relu'))
    model.add(tfkl.MaxPooling2D(pool_size=pool_size))

#aggiungere il layer di flatten
    model.add(tfkl.GlobalAveragePooling2D(name='gap'))

#aggiungere il layer fully connected
    model.add(tfkl.Dense(dense_units, activation='relu'))

#aggiungere il layer di dropout per ridurre l'overfitting
    model.add(tfkl.Dropout(dropout_rate))

#aggiungere il layer di output
    model.add(tfkl.Dense(dense_units, activation='softmax'))

#inserire l'Adam optimizer
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
model = build_model(input, 3, 2, 32, 32, 0.2)
history = model.fit(train_normalized, validation_data=val_normalized, epochs=100,callbacks=[early_stopping])

loss, accuracy = model.evaluate(val_normalized)
print('Test accuracy:', accuracy)