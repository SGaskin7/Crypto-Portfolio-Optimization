import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout, QInputDialog
from PyQt5.QtGui import QPixmap
import os


def get_img(num):
    img_path1 = "C:\\Users\\samga\\OneDrive\\Capstone\\GUI\\{title}".format(title='Bar-Graph---reg.png')
    img_path2 = "C:\\Users\\samga\\OneDrive\\Capstone\\GUI\\{title}".format(title='Bar-Graph---turnover.png')
    img = [img_path1,img_path2]
    return img[num]

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.img_path = 0
        
        self.MVO_Button = QPushButton("MVO",self)
        self.MVO_Button.clicked.connect(self.update)
        
        #MVO_Button = QPushButton("MVO")
        #MVO_Button.clicked.connect(self.update)

        self.CVAR_Button = QPushButton("CVAR",self)
        self.UI()#MVO_Button)
        

    def UI(self):#,MVO_Button):
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.label = QLabel(self)
        layout.addWidget(self.label)
        layout.addWidget(self.MVO_Button)#MVO_Button)
        layout.addWidget(self.CVAR_Button)

        self.setWindowTitle('Crypto Portfolio')
        self.showimage()
        self.show()

    def showimage(self):
        num = self.img_path
        path = get_img(num)
        pixmap = QPixmap(path)
        self.label.setPixmap(pixmap)
    
    def update(self):
        print('Button Clicked')
        if self.img_path == 0:
            self.img_path = 1
        else:
            self.img_path = 0
        self.showimage()



def main():

    application = QApplication(sys.argv)

    widget = App()
    widget.resize(350,150)
    widget.setWindowTitle('TITLE!')

    widget.show()

    sys.exit(application.exec_())

if __name__ == '__main__':
    main()


#if __name__ == '__main__':
#    app = QApplication(sys.argv)
#    ex = App()
#    sys.exit(app.exec_())

