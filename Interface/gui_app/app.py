import sys

import numpy as np
from PyQt6.QtGui import QImage, QPainter, QPen
from PyQt6.QtCore import Qt, QSize, QTimer, QFile, QTextStream
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
)
from modelNN.model import DigitPredictor
from config import WEIGHT_PATH


class DrawingCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)
        self.real_size = 28

        self.image = QImage(QSize(self.real_size, self.real_size), QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)

        self.drawing = False
        self.last_point = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = self.scale_point(event.pos())

    def mouseMoveEvent(self, event):
        if self.drawing and self.last_point:
            painter = QPainter(self.image)
            painter.setPen(
                QPen(
                    Qt.GlobalColor.black,
                    2,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            current_point = self.scale_point(event.pos())
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def clear(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def scale_point(self, point):
        x = max(0, min(self.real_size - 1, int(point.x() * self.real_size / self.width())))
        y = max(0, min(self.real_size - 1, int(point.y() * self.real_size / self.height())))
        return point.__class__(x, y)

    def get_image_array(self):
        ptr = self.image.bits()
        ptr.setsize(self.image.sizeInBytes())  # type: ignore
        arr = np.frombuffer(ptr, np.uint8).reshape(self.real_size, self.real_size, 4)  # type: ignore
        return arr[:, :, 0].astype(np.float32) / 255.0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = DigitPredictor(WEIGHT_PATH)

        self.symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.init_ui()
        self.load_styles()

    def load_styles(self):
        file = QFile("styles.css")

        if file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
            stream = QTextStream(file)
            self.setStyleSheet(stream.readAll())
            file.close()

    def init_ui(self):
        self.setWindowTitle("GUI CNN Recognizer")
        self.setFixedSize(900, 500)

        self.canvas = DrawingCanvas()
        
        self.madeBy = QLabel("psychea <3")

        self.clear_button = QPushButton("Clear canvas")
        self.clear_button.clicked.connect(self.canvas.clear)
        self.clear_button.setFixedWidth(150)

        self.prob_labels = []
        self.prob_bars = []
    
        probabilities_layout = QVBoxLayout()
        probabilities_layout.setSpacing(8)

        for symbol in self.symbols:
            hbox = QHBoxLayout()
            hbox.setSpacing(10)

            label = QLabel(f"{symbol}:")
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            prob_bar = QProgressBar()
            prob_bar.setRange(0, 100)
            prob_bar.setFixedHeight(20)
            prob_bar.setTextVisible(False)

            prob_label = QLabel("0%")
            prob_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            hbox.addWidget(label)
            hbox.addWidget(prob_bar)
            hbox.addWidget(prob_label)

            probabilities_layout.addLayout(hbox)
            self.prob_labels.append(prob_label)
            self.prob_bars.append(prob_bar)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(20)
        left_panel.setContentsMargins(20, 20, 20, 20)
        left_panel.addWidget(self.canvas)
        left_panel.addWidget(self.clear_button, 0, Qt.AlignmentFlag.AlignCenter)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(20)
        right_panel.setContentsMargins(20, 20, 20, 20)
        right_panel.addLayout(probabilities_layout)
        right_panel.addWidget(self.madeBy, 0, Qt.AlignmentFlag.AlignRight)
        right_panel.addStretch()

        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.addLayout(left_panel, stretch=2)
        main_layout.addLayout(right_panel, stretch=3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_prediction)
        self.timer.start(200) # update rate

    def update_prediction(self):
        image_array = self.canvas.get_image_array()
        probs, predicted_idx = self.predictor.predict(image_array)

        for i, (prob, label, bar) in enumerate(zip(probs, self.prob_labels, self.prob_bars)):
            percent = int(round(prob * 100))
            label.setText(f"{percent}%")

            if i == predicted_idx:
                bar.setStyleSheet(
                    """
                    QProgressBar::chunk {
                        background: #4CAF50;
                        border-radius: 3px;
                    }
                    """
                )
            else:
                bar.setStyleSheet(
                    """
                    QProgressBar::chunk {
                        background: #E0E0E0;
                        border-radius: 3px;
                    }
                    """
                )

            bar.setValue(percent)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
