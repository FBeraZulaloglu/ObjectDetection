# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qtDeneme.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from source.RunModel import Predict
from PyQt5.QtWidgets import QMainWindow,QMessageBox
from PyQt5.QtGui import QIcon
import cvlib
from cvlib.object_detection import draw_bbox
import cv2

import os

class Ui_MainWindow(QMainWindow):
    def setupUi(self,MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        #to show the image that we wanna guess
        self.photo = QtWidgets.QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(40, 70, 1200, 800))
        self.photo.setText("")
        self.photo.setPixmap(QtGui.QPixmap(r"C:\Users\faruk\PycharmProjects\ObjectDetectionProject\source\Objects\apple\apple.jpg"))
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")
        #to get the image from the selected path
        self.get_image = QtWidgets.QPushButton(self.centralwidget)
        self.get_image.setGeometry(QtCore.QRect(1400, 150, 150, 50))
        self.get_image.setText("SELECT YOUR IMAGE")
        # to select path
        self.path_selection = QtWidgets.QFileDialog(self.centralwidget)
        self.path_selection.setObjectName("SELECT YOUR IMAGE")
        #to read path
        self.path = QtWidgets.QTextEdit(self.centralwidget)
        self.path.setGeometry(QtCore.QRect(1400, 70, 350, 40))
        self.path.setObjectName("path")

        #JUST A TITLE
        self.result = QtWidgets.QLabel(self.centralwidget)
        self.result.setGeometry(QtCore.QRect(1400, 280, 400, 100))
        font = QtGui.QFont()
        font.setFamily("Perpetua")
        font.setPointSize(36)
        self.result.setFont(font)
        self.result.setObjectName("result")

        # shows the result
        self.prediction = QtWidgets.QLabel(self.centralwidget)
        self.prediction.setGeometry(QtCore.QRect(1450, 350, 300, 500))
        font = QtGui.QFont()
        font.setFamily("Perpetua")
        font.setPointSize(24)
        self.prediction.setFont(font)

        #guess the image from model
        self.guessButton = QtWidgets.QPushButton(self.centralwidget)
        self.guessButton.setGeometry(QtCore.QRect(1600, 150, 150, 50))
        self.guessButton.setObjectName("guessButton")
        # not using !!!
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1120, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        #to predict first image
        self.path.setText(r"C:\Users\faruk\PycharmProjects\ObjectDetectionProject\source\Objects\apple\apple.jpg")
        #predictions,label = Predict.predictImage(Predict,self.path.toPlainText())

        #buttons and actions
        self.guessButton.clicked.connect(self.guess_clicked)
        self.get_image.clicked.connect(self.load_image)
        #config the window
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.result.setText(_translate("MainWindow", "THE RESULT IS ..."))
        self.guessButton.setText(_translate("MainWindow", "GUESS THE IMAGE"))

    def guess_clicked(self):
        #Message Box
        self.prediction.setText("PROCESSING ...")
        message = QMessageBox.information(self,"GUESS","GUESS THE IMAGE ?",QMessageBox.Ok,QMessageBox.Cancel)
        if message == QMessageBox.Ok:
            try:
                predictions, label = Predict.predictImage(Predict, self.path.toPlainText())

                if not os.path.exists(self.path.toPlainText()):
                    print("THE PATH IS NOT SUITABLE")
                else:
                    self.photo.setPixmap(QtGui.QPixmap(self.path.toPlainText()))
                """
                counter = 0
                for p in predictions[0]:
                    self.p = QtWidgets.QLabel(self.centralwidget)
                    self.p.setGeometry(QtCore.QRect(1400, 280, 400, 300))
                    font = QtGui.QFont()
                    font.setFamily("Perpetua")
                    font.setPointSize(36)
                    self.p.setFont(font)
                    self.p.setObjectName("result")
                    x = "{}: {:.2f}%".format(label[counter], p * 100)
                    self.p.setText(x)
                    counter += 1
                """
                i = predictions.argmax(axis=1)[0]
                self.prediction.setText("{}: {:.2f}%".format(label[i], predictions[0][i] * 100))
            except Exception as e:
                self.prediction.setText("THE IMAGE COULDNT \n GUESSED")
                print("THE IMAGE HAS NOT FOUND")
                print(e)
        else:
            self.prediction.setText("")



    def load_image(self):
        path, _ = self.path_selection.getOpenFileName(self, "Open Image", r"C:\Users\faruk\Desktop", "Image Files(*.jpg *.png)")
        self.path_selection.setGeometry(QtCore.QRect(700, 140, 200, 40))
        self.photo.setPixmap(QtGui.QPixmap(path))
        self.path.setText(path)




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
