from math import pi
from scipy.constants import c
from random import *
from scipy.fft import rfft, rfftfreq
from PyQt5.QtCore import pyqtSlot as slot
from PyQt5.QtWidgets import QFileDialog, QAbstractItemView
from PyQt5 import QtWidgets, QtCore, uic
import pyqtgraph as pg
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import sip
import os

Design, _ = uic.loadUiType('Spectra.ui')

class ExampleApp(QtWidgets.QMainWindow, Design):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.getFile)
        self.range_select.clicked.connect(self.catgraph)
        self.pushButton_2.clicked.connect(self.plot_selected_region)
        self.pushButton_3.clicked.connect(self.get_n)
        self.chooseRange.clicked.connect(self.choose_range)

        self.filename = ''
        self.dftot = pd.DataFrame()
        # self.actionExit.triggered.connect(MainWindow.close)
        self.actionOpen.triggered.connect(self.getFile)
        self.canvas_Et.addLegend()
        self.canvas1_3.addLegend()
        self.canvas1_4.addLegend()

        # Get selected file in listWidget
        self.listWidget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.listWidget.itemSelectionChanged.connect(self.listWidget_on_change)

        # Dict for storage file name and Dataframe
        self.nameOfFile = {}

        # Create dict Dataframe

        self.dataframe_collection = {}
        self.selected_range = {}


        self.color = {}

        self.lr = pg.LinearRegionItem([10,50])
        self.canvas_Et.sigXRangeChanged.connect(self.regionUpdated)
        self.lr.sigRegionChanged.connect(self.regionUpdated)

        #self.amp.valueChanged.connect(self.regionUpdated)
        self.lr.sigRegionChangeFinished.connect(self.fin)



        # Dict for curve
        self.fft = {}
        self.freq_dict = {}
        self.unwrap = {}
        self.phase = {}

    def region(self):
        for item in [item.text() for item in self.listWidget.selectedItems()]:
            lo, hi = self.lr.getRegion()
            self.amp.setValue(hi * 10e11)
            print(lo,hi)
            self.selected_range[item] = [
                self.dataframe_collection[item][(self.dataframe_collection[item]['X_Value'] <= hi) & (
                        self.dataframe_collection[item]['X_Value'] >= lo)].iloc[0].name,
                self.dataframe_collection[item][(self.dataframe_collection[item]['X_Value'] <= hi) & (
                        self.dataframe_collection[item]['X_Value'] >= lo)].iloc[-1].name
            ]
            l, r = self.selected_range[item]


            self.lr.setRegion([self.dataframe_collection[item]['X_Value'][l], self.dataframe_collection[item]['X_Value'][r + 1]])


    def fin(self):
        print('g')


    def choose_range(self):
        for item in [item.text() for item in self.listWidget.selectedItems()]:
            if not bool(self.selected_range):
                self.lr = pg.LinearRegionItem(
                    [self.dataframe_collection[item]['X_Value'][0], self.dataframe_collection[item]['X_Value'][50]])
                self.amp.setValue(self.dataframe_collection[item]['X_Value'][50] * 10e11)

            self.canvas_Et.addItem(self.lr)


    def regionUpdated(self):
        pass


    def catgraph(self):
        if self.lr.getRegion():
            for item in [item.text() for item in self.listWidget.selectedItems()]:
                lo, hi = self.lr.getRegion()

                self.selected_range[item] = [
                    self.dataframe_collection[item][(self.dataframe_collection[item]['X_Value'] <= hi) & (
                                self.dataframe_collection[item]['X_Value'] >= lo)].iloc[0].name,
                    self.dataframe_collection[item][(self.dataframe_collection[item]['X_Value'] <= hi) & (
                            self.dataframe_collection[item]['X_Value'] >= lo)].iloc[-1].name
                ]



    def plot_selected_region(self):
        self.canvas_Et.clear()
        self.canvas1_3.clear()
        self.canvas1_4.clear()
        self.canvas1_5.clear()
        for item in [item.text() for item in self.listWidget.selectedItems()]:
            if item in self.selected_range:
                lo, hi = self.selected_range[item]
                print(lo, hi, hi-lo)
                print(len(self.dataframe_collection[item]['X_Value'].iloc[lo:hi].values))
                self.lr = pg.LinearRegionItem(
                    [self.dataframe_collection[item]['X_Value'][lo], self.dataframe_collection[item]['X_Value'][hi+1]])
                a = self.dataframe_collection[item]['X_Value'].iloc[lo:hi+1].values
                b = self.dataframe_collection[item]['scan'].iloc[lo:hi+1].values
                self.canvas_Et.plot(a, b,
                                   pen=self.color[item], name=item)

                self.ftt_E_1 = np.abs(np.fft.fft(b))
                #self.fft[item] = self.ftt_E_1

                self.freq = np.fft.fftfreq(len(self.ftt_E_1), self.dataframe_collection[item]['X_Value'][lo+1] -
                                           self.dataframe_collection[item]['X_Value'][lo])

                self.curve_ftt_1 = self.canvas1_3.plot(self.freq[:int(len(self.freq) / 2)],
                                                       self.ftt_E_1[:int(len(self.freq) / 2)], pen=self.color[item],
                                                       name=item).setLogMode(False, True)

                self.phase_ = np.angle(np.fft.fft(b))

                self.curve_phase_1 = self.canvas1_4.plot(self.phase_[:len(self.freq) // 2], pen=self.color[item])

                self.unwrap_ = np.unwrap(self.phase_)
                self.curve_unwrap_phase_1 = self.canvas1_4.plot(self.unwrap_[:len(self.freq) // 2],


    @slot(float)
    def on_amp_valueChanged(self,x):
        self.region()

    def listWidget_on_change(self):
        self.canvas_Et.clear()
        self.canvas1_3.clear()
        self.canvas1_4.clear()
        self.canvas1_5.clear()
        self.Update()

    def Update(self):

        for item in [item.text() for item in self.listWidget.selectedItems()]:
            # Кривая зависимости от времени lr = pg.LinearRegionItem([10, 40])
            self.canvas_Et.plot(self.dataframe_collection[item]['X_Value'], self.dataframe_collection[item]['scan'],
                                pen=self.color[item], name=item)

            self.ftt_E_1 = np.abs(np.fft.fft(self.dataframe_collection[item]['scan']))
            self.fft[item] = self.ftt_E_1

            self.freq = np.fft.fftfreq(len(self.ftt_E_1), self.dataframe_collection[item]['X_Value'][1] - self.dataframe_collection[item]['X_Value'][0])
            self.freq_dict[item] = self.freq


            self.curve_ftt_1 = self.canvas1_3.plot(self.freq[:int(len(self.freq) / 2)],
                                                   self.ftt_E_1[:int(len(self.freq) / 2)], pen=self.color[item],name=item).setLogMode(False, True)

            self.phase[item] = np.angle(np.fft.fft(self.dataframe_collection[item]['scan']))

            self.curve_phase_1 = self.canvas1_4.plot(self.phase[item][:len(self.freq) // 2], pen=self.color[item])

            self.unwrap[item] = np.unwrap(self.phase[item])
            self.curve_unwrap_phase_1 = self.canvas1_4.plot(self.unwrap[item][:len(self.freq) // 2], pen=self.color[item], name=item)





    def get_n(self):
        if len([item.text() for item in self.listWidget.selectedItems()]) >= 2:
            sample = [item.text() for item in self.listWidget.selectedItems()][0]
            referense = [item.text() for item in self.listWidget.selectedItems()][1]

            if (sample and referense) in self.selected_range:
                lo_s, hi_s = self.selected_range[sample]
                lo_r, hi_r = self.selected_range[referense]

                a = self.dataframe_collection[sample].iloc[lo_s:hi_s+1]
                b = self.dataframe_collection[referense].iloc[lo_r:hi_r+1]
            else:

                diff = self.dataframe_collection[sample].shape[0] - self.dataframe_collection[referense].shape[0]
                if diff > 0:
                    a = self.dataframe_collection[sample].iloc[:-diff]
                    b = self.dataframe_collection[referense]
                else:
                    a = self.dataframe_collection[sample]
                    b = self.dataframe_collection[referense].iloc[:-diff]


            ang_1 = np.angle(np.fft.fft(a['scan']))
            ang_2 = np.angle(np.fft.fft(b['scan']))
            freq_1 = np.fft.fftfreq(len(ang_1), a['X_Value'][1] - a['X_Value'][0])
            ang_H = np.unwrap(ang_1) - np.unwrap(ang_2)
            from scipy.constants import c
            n = (1 - (c * ang_H[1:] / (freq_1[1:] * 2.08 / 1000 * 6.28)))
            self.canvas1_6.plot(freq_1[1:len(freq_1) // 2], n[1:len(freq_1) // 2])

    def getFile(self):
        self.filename = QFileDialog.getOpenFileNames(filter="lvm (*.lvm);; csv (*.csv)")[0]
        self.readData()

    def readData(self):
        for item in self.filename:
            base_name = os.path.basename(item)
            self.color[base_name] = (randint(1, 256), randint(1, 256), randint(1, 256))
            self.listWidget.addItem(base_name)
            if os.path.splitext(base_name)[1] == '.csv':
                self.dataframe_collection[base_name] = pd.read_csv(item, encoding='utf-8').fillna(0)
            elif os.path.splitext(base_name)[1] == '.lvm':
                self.dataframe_collection[base_name] = pd.read_csv(item, sep='\t', skiprows=21, decimal=',')
            self.Update()

if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение