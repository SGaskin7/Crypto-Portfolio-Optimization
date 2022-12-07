import sys
from PyQt5.QtWidgets import QApplication,QLineEdit, QWidget, QCalendarWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout, QInputDialog, QComboBox, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDate
import os
import time


import sys
from GUIGraph import GUI_Chart
sys.path.insert(1, 'FactorResearch/backtesting')
from Backtester import Backtester

from PyQt5.QtWidgets import (
    #QApplication,
    QCheckBox,
    QTabWidget,
    QVBoxLayout,
    QWidget
)

def get_img(num,state, first=True):
    if not first:
        backtester = Backtester(predictor=state['Predictor'], optimizer=state['Optimizer'], alpha=state['alpha'], max_weight=state['Weight'], l=state['lambda_val'])
        backtester.LoadData()
        json_with_data = backtester.RunBacktesting(state['start'], state['end'])
        GUI_Chart(json_with_data, state)

        # print(json_with_data)

    #SAVE TAKE TO PNG
    #img_path1 = "C:\\Users\\samga\\OneDrive\\Capstone\\GUI\\GUI.png"
    #img_path2 = "C:\\Users\\samga\\OneDrive\\Capstone\\GUI\\GUI.png"

    img_path1 = 'GUI/GUI.png'
    img_path2 = 'GUI/GUI.png'

    img = [img_path1,img_path2]

    #SAVE MORE DATA IN JSON

    return img[num]


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto Portfolio Optimization Tool")
        self.resize(670, 110)

        #---- Current State ----
        self.state = {'Optimizer':'MVO','Predictor':'Price MA','Weight':0,'lambda_val':40,'alpha':0.05,'start':'2020-10-07','end':'2021-06-28'}
        self.counter = 0

        # Create a top-level layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        # Create the tab widget with two tabs
        tabs = QTabWidget()

        MVO_UI = self.New_MVO_UI()
        VaR_UI = self.New_VaR_UI()

        tabs.addTab(MVO_UI, "MVO")
        tabs.addTab(VaR_UI, "CVaR")
        self.layout.addWidget(tabs,1,0)

        #Second Tab WIdget
        calenders = QTabWidget()
        calenders.addTab(self.start_dateUI(), "Start Date (Earliest - 2020 Oct 1)")
        calenders.addTab(self.end_dateUI(), "End Date (Latest - 2021 July 05)")
        self.layout.addWidget(calenders,1,1)

        self.IMAGE = QLabel(self)
        self.layout.addWidget(self.IMAGE,0,0,1,2)

        self.loading = QLabel(self)
        self.pixmap_loading = QPixmap('GUI/loading.png')
        self.loading.setPixmap(self.pixmap_loading)

        self.showimage2(first=True)

    def showLoading(self):
        self.layout.removeWidget(self.IMAGE)
        self.layout.addWidget(self.loading,0,0,1,2)

    def showimage2(self, first=None):
        if self.counter == 0:
            self.counter = 1
        else:
            self.counter = 0

        # try:
        print('started')
        if not first:
            self.IMAGE.setScaledContents(True)
            self.IMAGE.setPixmap(QPixmap('GUI/loading.png'))

        path = get_img(self.counter,self.state, first)
        pixmap = QPixmap(path)
        self.IMAGE.setPixmap(pixmap)
        print('started')
        # except Exception as e:
        #     print(e)
        #     msg = QMessageBox()
        #     msg.setIcon(QMessageBox.Information)
        #     msg.setText("This is a message box")
        #     msg.setInformativeText("This is additional information")
        #     msg.setWindowTitle("MessageBox demo")
        #     msg.setDetailedText("The details are as follows:")

    
    def New_MVO_UI(self):
        
        generalTab = QWidget()
        layout = QGridLayout()

        Opt_label = QLabel('MVO Type')
        layout.addWidget(Opt_label,0,0)

        self.MVO_dropdown = QComboBox()
        self.MVO_dropdown.addItems(['MVO','Robust MVO'])
        layout.addWidget(self.MVO_dropdown,0,1)

        pred_label = QLabel('Predictor')
        layout.addWidget(pred_label,1,0)

        self.Predictor_dropdown = QComboBox()
        self.Predictor_dropdown.addItems(['Price MA', 'Factor Model', 'Decision Tree'])
        layout.addWidget(self.Predictor_dropdown,1,1)

        weight_label = QLabel('Max Weight Per Asset (0-1)')
        
        layout.addWidget(weight_label,2,0)

        self.weight = QLineEdit()
        self.weight.setText('1')
        layout.addWidget(self.weight,2,2)

        lambda_label = QLabel('Lambda (Quadratic Factor)')
        layout.addWidget(lambda_label,3,0)

        self.lambda_val = QLineEdit()
        self.lambda_val.setText('1')
        layout.addWidget(self.lambda_val,3,3)
        
        self.dates_check_MVO = QCheckBox('Use Entire Date Range')
        layout.addWidget(self.dates_check_MVO,4,0)

        self.MVO_simulate = QPushButton('Run Simulation')
        self.MVO_simulate.clicked.connect(self.update_MVO)
        layout.addWidget(self.MVO_simulate,5,0,1,2)

        #lambda_val = QLabel('Linear Factor')
        #layout.addWidget(weight_label,4,0)

        #lambda_val = QLineEdit()
        #layout.addWidget(lambda_val,4,4)
        
        generalTab.setLayout(layout)
        return generalTab


    def New_VaR_UI(self):

        generalTab = QWidget()
        layout = QGridLayout()

        Opt_label = QLabel('Simulation Method')
        layout.addWidget(Opt_label,0,0)

        self.simulate_dropdown = QComboBox()
        self.simulate_dropdown.addItems(['Historical','Normal','ARIMA-Garch'])
        layout.addWidget(self.simulate_dropdown,0,1)

        pred_label = QLabel('Alpha Level (Default is 0.05)')
        layout.addWidget(pred_label,1,0)

        self.alpha_val = QLineEdit()
        self.alpha_val.setText('0.05')
        layout.addWidget(self.alpha_val,1,1)
        
        self.dates_check_VaR = QCheckBox('Use Entire Date Range')
        layout.addWidget(self.dates_check_VaR,2,0)

        self.CVaR_simulate = QPushButton('Run Simulation')
        self.CVaR_simulate.clicked.connect(self.update_CVaR)
        layout.addWidget(self.CVaR_simulate,3,0,1,2)

        
        generalTab.setLayout(layout)
        return generalTab


    def start_dateUI(self):
        self.start_date = QCalendarWidget()
        self.start_date.setMinimumDate(QDate(2020,10,1))
        self.start_date.setMaximumDate(QDate(2021,7,5))
        self.start_date.setSelectedDate(QDate(2020,10,2))
        #layout = QVBoxLayout()
        #layout.addWidget(QCalendarWidget)
        #start_date.setLayout(layout)
        return self.start_date

    def end_dateUI(self):
        self.end_date = QCalendarWidget()
        self.end_date.setMinimumDate(QDate(2020,10,1))
        self.end_date.setMaximumDate(QDate(2021,7,5))
        self.end_date.setSelectedDate(QDate(2021,7,5))

        #layout = QVBoxLayout()
        #layout.addWidget(QCalendarWidget)
        #end_date.setLayout(layout)
        return self.end_date


    def calender_UI(self):
        Calender = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QCalendarWidget())
        Calender.setLayout(layout)
        return Calender
    
    def update_MVO(self):
        self.state['Optimizer'] = ['MVO','RMVO'][self.MVO_dropdown.currentIndex()]
        self.state['Predictor'] = ['Price MA', 'Factor Model', 'Decision Tree'][self.Predictor_dropdown.currentIndex()]
        self.state['Weight'] = float(self.weight.text())
        self.state['lambda_val'] = float(self.lambda_val.text())
        self.state['start'] = str(self.start_date.selectedDate().toString('yyyy-MM-dd'))
        self.state['end'] = str(self.end_date.selectedDate().toString('yyyy-MM-dd'))
        if self.dates_check_MVO.isChecked():
            self.state['start'] = '2020-10-07'
            self.state['end'] = '2021-07-05'

        self.showimage2(first=False)
        return True
    
    def update_CVaR(self):
        self.state['Predictor'] = 'CVaR'
        self.state['Optimizer'] = ['Historical','Normal','ARIMA-GARCH'][self.simulate_dropdown.currentIndex()]
        self.state['alpha'] = float(self.alpha_val.text())
        self.state['start'] = str(self.start_date.selectedDate().toString('yyyy-MM-dd'))
        self.state['end'] = str(self.end_date.selectedDate().toString('yyyy-MM-dd'))
        if self.dates_check_VaR.isChecked():
            self.state['start'] = '2020-10-07'
            self.state['end'] = '2021-07-05'

        self.showimage2(first=False)
        return True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())