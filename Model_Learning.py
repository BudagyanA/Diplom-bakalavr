import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, save_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import time

# Загрузка и предобработка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация и изменение размера изображений
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Создание модели сверточной нейронной сети
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Ранняя остановка
early_stopping = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)

# Обучение модели
start_time = time.time()
model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test), callbacks=[early_stopping])
end_time = time.time()

# Оценка модели на тестовом наборе данных
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Точность на тестовом наборе данных:", test_accuracy)
print("Потери на тестовом наборе данных:", test_loss)

# Время обучения
training_time = end_time - start_time
print("Время обучения:", training_time, "секунд")

# Сохранение модели
model.save("model.h5")
print("Модель сохранена, как model.h5")
time.sleep(10)
