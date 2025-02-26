# %%
from keras import layers
from keras import models
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# %%
import os

# %%
def plot_history(hst):
    plt.figure(figsize=(13, 4))

    plt.subplot(1, 2, 1)
    plt.plot(hst.history['loss'], label='train')
    plt.plot(hst.history['val_loss'], label='test')
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot([round(100*e, 2) for e in hst.history['acc']], label='train')
    plt.plot([round(100*e, 2) for e in hst.history['val_acc']], label='test')
    plt.title('Accuracy')    

    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# Частина 1. Побудова архітектури згорткової нейронної мережі

# %%

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),

    layers.Dense(10, activation="softmax")
])


# %%
model.summary()

# %%

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# %%
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# %%
model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# %%
# creating callback that saves weights only during training
checkpoint_path = r"C:\Users\Anastasiia\Documents\PythonProjects\Models\Model_1_ckpt.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


# %%
history  =  model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=5, batch_size=64, callbacks=[cp_callback])

# %%
plot_history(history)

# %% [markdown]
# Частина 2. Використання передвиборних мереж. Виділення ознак

# %%
import cv2

train_images_3D = np.zeros((60000, 32, 32, 3), dtype=train_images.dtype)
test_images_3D = np.zeros((10000, 32, 32, 3), dtype=train_images.dtype)

for i in range(train_images.shape[0]):
    # Resize the single image (28, 28, 1) to (224, 224, 3)
    resized_TR_image = cv2.resize(train_images[i, :, :, 0], (32, 32), interpolation=cv2.INTER_LINEAR)
    train_images_3D[i] = np.repeat(resized_TR_image[:, :, np.newaxis], 3, axis=2)
    if i < 10000:
        resized_T_image = cv2.resize(test_images[i, :, :, 0], (32, 32), interpolation=cv2.INTER_LINEAR)
        test_images_3D[i] = np.repeat(resized_T_image[:, :, np.newaxis], 3, axis=2)
    




# %%
batch_size = 1000
steps_per_epoch = len(train_images_3D)//batch_size
epochs = 50
validation_steps = len(test_images_3D)//batch_size

# %%


train_datagen =  tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_datagen.fit(train_images_3D)
test_datagen.fit(test_images_3D)

train_generator = train_datagen.flow(train_images_3D, train_labels, batch_size=batch_size)

validation_generator = test_datagen.flow(test_images_3D, test_labels, batch_size=batch_size)

# %%
from keras.applications.vgg16 import VGG16

conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(32,32,3))

conv_base.trainable = False

model = models.Sequential([
   conv_base,
   layers.Flatten(),
   layers.Dense(256, activation="relu"),
   layers.Dense(10, activation="softmax"),
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-5),
    metrics=["acc"]
)

model.summary()

# %%
# creating callback that saves weights only during training
checkpoint_path_1 = r"C:\Users\Anastasiia\Documents\PythonProjects\Models\Model_2_ckpt.weights.h5"
checkpoint_dir_1 = os.path.dirname(checkpoint_path_1)

cp_callback_1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_1,
                                                 save_weights_only=True,
                                                 verbose=1)


# %%
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 3, 
    verbose = 1
)

history_1 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping,cp_callback_1]
)



# %%
plot_history(history_1)

# %% [markdown]
# Частина 2. Використання передвиборних мереж. Донавчання

# %%
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(32,32,3))

# %%
set_trainable = False
for layer in conv_base.layers:
    if layer.name == "block4_conv2":
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# %%
model = models.Sequential([
   conv_base,
   layers.Flatten(),
   layers.Dense(256, activation="relu"),
   layers.Dense(10, activation="softmax"),
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-5),
    metrics=["acc"]
)

model.summary()

# %%
# creating callback that saves weights only during training
checkpoint_path_2 = r"C:\Users\Anastasiia\Documents\PythonProjects\Models\Model_3_ckpt.weights.h5"
checkpoint_dir_2 = os.path.dirname(checkpoint_path_2)

cp_callback_2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_2,
                                                 save_weights_only=True,
                                                 verbose=1)

# %%
history_2 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping,cp_callback_2]
)

# %%
plot_history(history_2)

# %%
train_images_3D.shape


