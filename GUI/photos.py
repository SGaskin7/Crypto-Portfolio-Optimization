import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Document Analysis'
        self.left = 30
        self.top = 30
        self.width = 640
        self.height = 480
        self.imagenumber=0
        
        #self.ok_button = QPushButton("OK",self)
        
        self.initUI()
        

    def keyPressEvent(self, event):
        key=event.key()
        if key==Qt.Key_Right:
            self.imagenumber=self.imagenumber+1
            self.showimage(self.imagenumber)
            # self.show()
        else:
            super(self).keyPressEvent(event)
        print(imgnum(self.imagenumber))

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.label = QLabel(self)
        layout.addWidget(self.label)
        #layout.addWidget(self.ok_button)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.showimage(0)
        self.show()

    def showimage(self,imagenumber):
        # label = QLabel(self)

        directory = "C:\\Users\\samga\\OneDrive\\Capstone\\GUI"
        imagelist = os.listdir(directory)
        pixmap = QPixmap(directory + '\\' + imagelist[imagenumber])

        # label.setPixmap(pixmap)
        self.label.setPixmap(pixmap)
        self.resize(pixmap.width() + 500, pixmap.height())
        # self.show()

def imgnum(number):
    return str(number)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())