import sys
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt6 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
sys.path.insert(1,r'C:\Users\samga\OneDrive\Desktop\Charts')
import SGCharts

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=20, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.set_title('Testing')
        sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        sc.axes.scatter([0,1,2,3,4],[1,2,3,4,5])
        self.setCentralWidget(sc)

        self.show()




app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec()