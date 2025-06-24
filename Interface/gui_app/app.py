import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QPushButton, QScrollArea)
from PyQt6.QtGui import QImage, QPainter, QPen, QColor, QTransform
from PyQt6.QtCore import Qt, QSize, QRect
import numpy as np


class DrawingBoard(QWidget):
    def __init__(self, width, height, scale_factor, parent=None):
        super().__init__(parent)
        self._width = width
        self._height = height
        
        self.scale_factor = scale_factor
        
        self.image = QImage(self._width, self._height, QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        
        self.drawing = False
        self.brush_size = 3
        self.brush_color = Qt.GlobalColor.black
        self.last_point = None

    def set_pixel(self, x, y, color):
        if 0 <= x < self._width and 0 <= y < self._height:
            self.image.setPixel(x, y, color.value)
            self.update()

    def pixel_coords(self, event_x, event_y):
        pixel_x = int(event_x / self.scale_factor)
        pixel_y = int(event_y / self.scale_factor)
        return pixel_x, pixel_y

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            pixel_x, pixel_y = self.pixel_coords(event.pos().x(), event.pos().y())
            self.set_pixel(pixel_x, pixel_y, self.brush_color)

    def mouseMoveEvent(self, event):
        if self.drawing:
            pixel_x, pixel_y = self.pixel_coords(event.pos().x(), event.pos().y())

            painter = QPainter(self.image)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

            last_pixel_x, last_pixel_y = self.pixel_coords(self.last_point.x(), self.last_point.y())

            painter.drawLine(last_pixel_x, last_pixel_y, pixel_x, pixel_y)
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.last_point = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.scale(self.scale_factor, self.scale_factor)
        painter.drawImage(0, 0, self.image)

    def clear_image(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def get_image_data(self):
        """get изображение в формате np.array"""
        buffer = self.image.constBits()
        img = np.frombuffer(buffer, np.uint8).reshape((self._height, self._width, 4))
        return img[:,:,:3]

    def sizeHint(self):
        return QSize(self._width * self.scale_factor, self._height * self.scale_factor)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drawing App with Scaled Board")

        self.image_width = 64
        self.image_height = 64
        self.scale_factor = 8

        self.drawing_board = DrawingBoard(self.image_width, self.image_height, self.scale_factor)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.drawing_board.clear_image)

        # хз хочу real time предикт
        # TODO надо кнопку убрать
        # но бля заебемся наверное хз можно для первых тестов оставить wwww

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)

        self.prediction_label = QLabel("Prediction: Jlox")

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.predict_button)
        controls_layout.addWidget(self.prediction_label)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.drawing_board)
        main_layout.addLayout(controls_layout)

        self.setLayout(main_layout)

    def predict(self):
        """get изображения и передачи в плюсыыыыы"""
        image_data = self.drawing_board.get_image_data()
        
        # тут отправка в предикшн
        
        prediction = self.fake_predict(image_data) 
        self.prediction_label.setText(f"Prediction: {prediction}")

    def fake_predict(self, image_data):
        """Функция предсказания"""
        # mozhno tak ostavit' xD
        return np.random.randint(0, 10)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())