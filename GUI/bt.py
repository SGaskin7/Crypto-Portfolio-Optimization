import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout, QInputDialog
from PyQt5.QtGui import QPixmap


class MainWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.button_show()

    def button_show(self):
        button = QPushButton('CLick Me',self)
        button.resize(200,60)
        button.clicked.connect(self.on_click)

    def on_click(self):
        print('CLICKED!!')

class ImageWidget(QWidget):

    def __init__(self):
        super().__init__()
        #self.show_image()
        
        self.ok_button = QPushButton("OK",self)
        self.cancel_button = QPushButton("Cancel!",self)
        
        
        self.horizontal_layout()

    def horizontal_layout(self):
        horizontal_bar = QHBoxLayout()
        horizontal_bar.addStretch(0)
        horizontal_bar.addWidget(self.ok_button)
        horizontal_bar.addWidget(self.cancel_button)
        self.setLayout(horizontal_bar)

    def show_image(self):
        image = QPixmap("Bar-Graph---reg.png")
        label = QLabel(self)
        label.setPixmap(image)
        


class ImgBut(QWidget):

    def __init__(self):
        super().__init__()

        self.button = QPushButton("OKK",self)

        vlay = QVBoxLayout(self)
        img = self.create_image()
        vlay.addLayout(img)
        vlay.addLayout(img)

    def create_image(self):
        vlay = QVBoxLayout
        image = QPixmap("Bar-Graph---reg.png")
        label = QLabel(self)
        label.setPixmap(image)
        vlay.addWidget(label)
        return vlay
        


class MainLayout(QWidget):

    def __init__(self):
        super().__init__()
        #self.the_two_widgets()
        #self.horizontal_layout()
        #self.verticle_layout()
        self.grid_layout()

    def the_two_widgets(self):
        self.ok_button = QPushButton("OK",self)
        self.cancel_button = QPushButton("Cancel!",self)

    def horizontal_layout(self):
        horizontal_bar = QHBoxLayout()
        horizontal_bar.addStretch(0)
        horizontal_bar.addWidget(self.ok_button)
        horizontal_bar.addWidget(self.cancel_button)
        self.setLayout(horizontal_bar)

    def verticle_layout(self):
        verticle_bar = QVBoxLayout()
        verticle_bar.addStretch(0)
        verticle_bar.addWidget(self.ok_button)
        verticle_bar.addWidget(self.cancel_button)
        self.setLayout(verticle_bar)


    def grid_layout(self):
        grid_box = QGridLayout()
        
        grid_data = ["OK", "CANCLL",
                    "Chose",'Exit']
        #grid_box.addWidget(QPushButton('1st'),0,0)
        #grid_box.addWidget(QPushButton('2ns'),0,1)
        
        #for data in grid_data:
        #    button = QPushButton(data, self)
        #    grid_box.addWidget(button)
        #    self.setLayout(grid_box)
        
        b0 = QPushButton('0',self)
        grid_box.addWidget(b0,0,0)
        #b1 = QPushButton('1',self)
        #grid_box.addWidget(b1,0,1)
        b2 = QPushButton('2',self)
        grid_box.addWidget(b2,1,0)
        b3 = QPushButton('3',self)
        grid_box.addWidget(b3,1,1)

        self.setLayout(grid_box)


class PopupWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        btn = QPushButton('Click Here',self)
        btn.move(20,20)
        btn.clicked.connect(self.show_dialog)

    def show_dialog(self):
        text, ok = QInputDialog.getText(self,'Sample DIALOG','Enter Your Name: ')


def main():

    application = QApplication(sys.argv)

    widget = MainLayout()
    widget.resize(350,150)
    widget.setWindowTitle('TITLE!')

    widget.show()

    sys.exit(application.exec_())

if __name__ == '__main__':
    main()