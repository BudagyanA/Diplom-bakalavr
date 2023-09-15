import cv2
import tensorflow as tf
from tkinter import Tk, Label, Button, filedialog
import numpy as np
from PIL import Image, ImageTk

model = tf.keras.models.load_model('model.h5')                                                         # Загрузка обученной модели

def select_image():                                                                                    # Открываем диалоговое окно для выбора изображения
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    image = cv2.imread(file_path)                                                                      # Загружаем изображение

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                                     # Преобразуем изображение в оттенки серого

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)                   # Применяем алгоритм бинаризации, чтобы получить черно-белое изображение

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)                 # Находим контуры на изображении

    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    recognized_digits = []                                                                             # Обходим контуры и распознаем каждую цифру

    for i, contour in enumerate(contours):

        (x, y, w, h) = cv2.boundingRect(contour)                                                       # Получаем ограничивающий прямоугольник для текущего контура

        digit = binary[y:y+h, x:x+w]                                                                   # Извлекаем цифру из ограничивающего прямоугольника

        height, width = digit.shape[:2]
        size = max(height, width)

        square_image = np.zeros((size, size), dtype=np.uint8)

        offset_x = (size - width) // 2                                                                 # Вычисляем смещение для вставки цифры по центру нового изображения
        offset_y = (size - height) // 2

        square_image[offset_y:offset_y+height, offset_x:offset_x+width] = digit                        # Вставляем цифру на новое изображение с черным фоном

        resized_image = cv2.resize(square_image, (28, 28), interpolation=cv2.INTER_AREA)

        digit_image = resized_image / 255.0                                                              # Нормализуем значения пикселей до диапазона [0, 1]

        digit_image = cv2.normalize(digit_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)       # Изменяем размерность цифры для подачи в модель

        digit = digit_image.reshape((28, 28, 1))

        prediction = model.predict(np.array([digit]))                                                  # Распознаем цифру с помощью модели
        recognized_digit = np.argmax(prediction)                                                       # Получаем распознанную цифру

        recognized_digits.append(recognized_digit)

    digits_text = ", ".join(str(digit) for digit in recognized_digits)                                 # Выводим распознанные цифры
    result_label.config(text="Распознанные цифры: " + digits_text)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                                                     # Отображаем выбранное изображение
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    image_label.config(image=image)
    image_label.image = image

root = Tk()                                                                                            # Создаем графический интерфейс с кнопкой для выбора изображения
root.title("Разделение и распознавание цифр")
root.geometry("400x300")

label = Label(root, text="Выберите изображение с цифрами")
label.pack(pady=10)

button = Button(root, text="Выбрать изображение", command=select_image)
button.pack()

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="")
result_label.pack(pady=10)

root.mainloop()
