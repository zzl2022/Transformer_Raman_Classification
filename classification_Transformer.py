# @File :classification_Transformer.py
# @Time :2022/8/18   15:11
# @Author : zhaozl
# @Describe :

from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from seaborn import heatmap
from sklearn.metrics import classification_report, confusion_matrix
import os

# %% make a folder

path =os.getcwd() + '\\result\\'
if not os.path.exists(path):
	os.makedirs(path)

# %% load the dataset

# Datasets can be obtained from the following URL:
# https://www.kaggle.com/datasets/sfran96/raman-spectroscopy-for-detecting-covid19
# Unzip the data set and put it in the "data" folder
data = pd.read_csv("data/covid_and_healthy_spectra.csv")

classes = data[data.columns[-1]].unique()
normal_mapping = dict(zip(classes, range(len(classes))))
label = data[data.columns[-1]].map(normal_mapping)
data[data.columns[-1]] = label

x_train, x_test, y_train, y_test = train_test_split(data.loc[:, :data.columns[-2]], data.loc[:, data.columns[-1]],
                                                    shuffle=True, random_state=7, test_size=0.2)

n_classes = len(classes)

# reshape the input data
x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

# %% classification with a Transformer model

# Transformer network architecture is based on Keras API, which refers to:
# https://keras.io/examples/timeseries/timeseries_classification_transformer/

# Build the model
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# Train and evaluate
input_shape = x_train.shape[1:]
model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
# model.summary()

callbacks = [
    # save model
    keras.callbacks.ModelCheckpoint(
        r"result\best_model_Transformer.h5", save_best_only=True, monitor="val_loss"
    ),
    # Adjust learning rate during training
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=4,
    callbacks=callbacks,
    )

metric = "sparse_categorical_accuracy"
plt.figure(figsize=(4, 3), dpi=300)
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.tight_layout()
plt.savefig(r'result\train_history.png')
plt.show()

# %% Evaluate the performance of the model after training

# Loading the trained model can avoid training time consumption when evaluating at any time.
model = keras.models.load_model(r"result\best_model_Transformer.h5")

test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=4, verbose=1)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

print(f"Classification report for test-dataset:\n{classification_report(np.argmax(model.predict(x_test, batch_size=4), axis=1), y_test)}")
# print(f"Classification report for train-dataset:\n{classification_report(np.argmax(model.predict(x_train, batch_size=4), axis=1), y_train)}")

plt.figure(figsize=(4, 3), dpi=300)
heatmap(confusion_matrix(np.argmax(model.predict(x_test, batch_size=4), axis=1), y_test), annot=True, fmt="g", yticklabels=["Healthy", "SARS-CoV-19"] ,xticklabels=["Healthy", "SARS-CoV-19"])
plt.title("Confusion matrix for test-dataset")
plt.xlabel("Predicted")
plt.ylabel("Expected")
plt.tight_layout()
plt.savefig(r'result\confusion_matrix.png')
plt.show()


