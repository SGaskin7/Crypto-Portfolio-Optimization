from PyQt6.QtWidgets import (
      QApplication, QVBoxLayout, QWidget, QLabel, QPushButton
)
from PyQt6.QtCore import Qt
import sys


import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=20, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

 
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(300, 250)
        self.setWindowTitle("CodersLegacy")
 
        layout = QVBoxLayout()
        self.setLayout(layout)
 
        self.label = QLabel("Old Text")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.adjustSize()
        layout.addWidget(self.label)
 
        button = QPushButton("Update Text")
        button.clicked.connect(self.update)
        layout.addWidget(button)
 
        button = QPushButton("Print Text")
        button.clicked.connect(self.get)
        layout.addWidget(button)
 
    def update(self):
        self.label.setText("New and Updated Text")

     
    def get(self):
        print(self.label.text())
         
 
app = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(app.exec())