from sklearn.model_selection import KFold
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras import layers as tfkl
from tensorflow import keras as tfk

# Caricamento del dataset
data = np.load("training_set.npz")
X, y = data['images'], data['labels']

# Parametri
inputShape = (X.shape[1], X.shape[2], 3)
kernel_size = 3
pool_size = 2
num_filters = 32
dense_units = 32
dropout_rate = 0.2
batch_size = 32
epochs = 200
n_splits = 5  # Numero di fold per la cross-validation

# KFold Cross-validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_accuracies = []


# Funzione per costruire il modello
def build_model(input_shape, kernel_size, pool_size, num_filters, dense_units, dropout_rate):
    model = tfk.Sequential()
    model.add(tfkl.Conv2D(num_filters, kernel_size, input_shape=input_shape, activation='relu'))
    model.add(tfkl.MaxPooling2D(pool_size=pool_size))
    model.add(tfkl.Conv2D(num_filters * 4, kernel_size, activation='relu'))
    model.add(tfkl.MaxPooling2D(pool_size=pool_size))
    model.add(tfkl.Conv2D(num_filters * 8, kernel_size, activation='relu'))
    model.add(tfkl.MaxPooling2D(pool_size=pool_size))
    model.add(tfkl.GlobalAveragePooling2D(name='gap'))
    model.add(tfkl.Dense(dense_units, activation='relu'))
    model.add(tfkl.Dropout(dropout_rate))
    model.add(tfkl.Dense(len(np.unique(y)), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Ciclo attraverso ciascun fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_splits}")

    # Suddivisione dei dati per il fold corrente
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Normalizzazione
    train_rescaled = ImageDataGenerator(rescale=1. / 255)
    val_rescaled = ImageDataGenerator(rescale=1. / 255)
    train_normalized = train_rescaled.flow(X_train, y_train, batch_size=batch_size)
    val_normalized = val_rescaled.flow(X_val, y_val, batch_size=batch_size)

    # Costruzione e addestramento del modello per il fold
    model = build_model(inputShape, kernel_size, pool_size, num_filters, dense_units, dropout_rate)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(train_normalized, validation_data=val_normalized, epochs=epochs, callbacks=[early_stopping])

    # Valutazione del modello sul validation set corrente
    loss, accuracy = model.evaluate(val_normalized)
    print(f"Fold {fold + 1} - Validation Accuracy: {accuracy}")
    fold_accuracies.append(accuracy)

# Media e deviazione standard delle accuratezze
print("\nCross-Validation Results:")
print(f"Accuracy per fold: {fold_accuracies}")
print(f"Mean Accuracy: {np.mean(fold_accuracies)}")
print(f"Standard Deviation: {np.std(fold_accuracies)}")

#Salva il modello
model_filename = 'CrossValidationModel'
model.save(model_filename)
del model

print("Modello salvato")