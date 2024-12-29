import os
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.models import save_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications.xception import Xception
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

# Các tham số về kích thước ảnh và batch size
batch_size = 40
width = 224
height = 224
channels = 3

# Đọc dữ liệu
train_data = tf.keras.utils.image_dataset_from_directory('/kaggle/input/sports-classification/train', label_mode='categorical', shuffle=False)
test_data = tf.keras.utils.image_dataset_from_directory('/kaggle/input/sports-classification/test', shuffle=False, label_mode='categorical')
validation_data = tf.keras.utils.image_dataset_from_directory('/kaggle/input/sports-classification/valid', label_mode='categorical', shuffle=False)

# Tạo datagen với augmentation
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    '/kaggle/input/sports-classification/train', 
    target_size=(224, 224), 
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,  
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    '/kaggle/input/sports-classification/test', 
    target_size=(224, 224), 
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False, 
    seed=42
)

# Khởi tạo mô hình EfficientNetB0
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# # Đặt các lớp trong base_model là không huấn luyện
# for layer in base_model.layers:
#     layer.trainable = False

# Thêm các lớp lên trên base_model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.25)(x)
x = Dense(256, activation='relu')(x)
x = Dense(100, activation='softmax')(x)  # Số lớp phân loại

# Định nghĩa mô hình hoàn chỉnh
model = Model(inputs=base_model.input, outputs=x)

# Hiển thị thông tin mô hình
model.summary()

# Tối ưu hóa và callbacks
optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-07)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7)

# Biên dịch mô hình
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
# Huấn luyện mô hình mà không sử dụng 'workers' và 'use_multiprocessing'
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=test_generator,  # Sử dụng test_generator làm validation
    callbacks=[reduce_lr]
)

# Lưu mô hình
model.save("sport.h5")
test_loss, test_acc = model.evaluate(test_generator, verbose=2)

# In kết quả
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(train_acc) + 1)

    # Plotting accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
