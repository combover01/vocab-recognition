# Form implementation generated from reading ui file 'test_training.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(30, 10, 741, 491))
        self.tabWidget.setTabBarAutoHide(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.l_loopCounter = QtWidgets.QLabel(parent=self.tab)
        self.l_loopCounter.setGeometry(QtCore.QRect(380, 180, 71, 21))
        self.l_loopCounter.setText("")
        self.l_loopCounter.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.l_loopCounter.setObjectName("l_loopCounter")
        self.label2_4 = QtWidgets.QLabel(parent=self.tab)
        self.label2_4.setGeometry(QtCore.QRect(290, 90, 121, 20))
        self.label2_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label2_4.setObjectName("label2_4")
        self.b_uploadrecordings = QtWidgets.QPushButton(parent=self.tab)
        self.b_uploadrecordings.setGeometry(QtCore.QRect(450, 160, 141, 31))
        self.b_uploadrecordings.setAutoFillBackground(False)
        self.b_uploadrecordings.setObjectName("b_uploadrecordings")
        self.label4_2 = QtWidgets.QLabel(parent=self.tab)
        self.label4_2.setGeometry(QtCore.QRect(40, 100, 231, 20))
        self.label4_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label4_2.setObjectName("label4_2")
        self.l_word_2 = QtWidgets.QLabel(parent=self.tab)
        self.l_word_2.setGeometry(QtCore.QRect(250, 110, 191, 20))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setUnderline(True)
        self.l_word_2.setFont(font)
        self.l_word_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.l_word_2.setObjectName("l_word_2")
        self.b_record_2 = QtWidgets.QPushButton(parent=self.tab)
        self.b_record_2.setEnabled(False)
        self.b_record_2.setGeometry(QtCore.QRect(290, 140, 120, 32))
        self.b_record_2.setObjectName("b_record_2")
        self.fp_wordbank_2 = QtWidgets.QLineEdit(parent=self.tab)
        self.fp_wordbank_2.setGeometry(QtCore.QRect(40, 130, 221, 21))
        self.fp_wordbank_2.setObjectName("fp_wordbank_2")
        self.label2_5 = QtWidgets.QLabel(parent=self.tab)
        self.label2_5.setGeometry(QtCore.QRect(210, 180, 171, 20))
        self.label2_5.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label2_5.setObjectName("label2_5")
        self.fp_prerecord = QtWidgets.QLineEdit(parent=self.tab)
        self.fp_prerecord.setGeometry(QtCore.QRect(450, 130, 251, 21))
        self.fp_prerecord.setObjectName("fp_prerecord")
        self.label3 = QtWidgets.QLabel(parent=self.tab)
        self.label3.setGeometry(QtCore.QRect(450, 100, 231, 20))
        self.label3.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label3.setObjectName("label3")
        self.b_process = QtWidgets.QPushButton(parent=self.tab)
        self.b_process.setEnabled(False)
        self.b_process.setGeometry(QtCore.QRect(110, 210, 551, 101))
        self.b_process.setAutoFillBackground(False)
        self.b_process.setObjectName("b_process")
        self.plainTextEdit_2 = QtWidgets.QPlainTextEdit(parent=self.tab)
        self.plainTextEdit_2.setGeometry(QtCore.QRect(240, 310, 301, 141))
        self.plainTextEdit_2.setReadOnly(True)
        self.plainTextEdit_2.setObjectName("plainTextEdit_2")
        self.label5 = QtWidgets.QLabel(parent=self.tab)
        self.label5.setGeometry(QtCore.QRect(260, 20, 261, 20))
        self.label5.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label5.setObjectName("label5")
        self.fp_1process = QtWidgets.QLineEdit(parent=self.tab)
        self.fp_1process.setGeometry(QtCore.QRect(260, 40, 251, 21))
        self.fp_1process.setObjectName("fp_1process")
        self.b_wordbank_2 = QtWidgets.QPushButton(parent=self.tab)
        self.b_wordbank_2.setGeometry(QtCore.QRect(40, 160, 141, 31))
        self.b_wordbank_2.setAutoFillBackground(False)
        self.b_wordbank_2.setObjectName("b_wordbank_2")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(parent=self.tab_2)
        self.plainTextEdit.setGeometry(QtCore.QRect(60, 210, 301, 101))
        self.plainTextEdit.setReadOnly(True)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.b_generatenew = QtWidgets.QPushButton(parent=self.tab_2)
        self.b_generatenew.setEnabled(False)
        self.b_generatenew.setGeometry(QtCore.QRect(430, 160, 221, 41))
        self.b_generatenew.setAutoFillBackground(False)
        self.b_generatenew.setObjectName("b_generatenew")
        self.label2_2 = QtWidgets.QLabel(parent=self.tab_2)
        self.label2_2.setGeometry(QtCore.QRect(120, 190, 171, 20))
        self.label2_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label2_2.setObjectName("label2_2")
        self.label4 = QtWidgets.QLabel(parent=self.tab_2)
        self.label4.setGeometry(QtCore.QRect(80, 90, 231, 20))
        self.label4.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label4.setObjectName("label4")
        self.l_spectrogram_accuracy = QtWidgets.QLabel(parent=self.tab_2)
        self.l_spectrogram_accuracy.setGeometry(QtCore.QRect(460, 270, 171, 20))
        self.l_spectrogram_accuracy.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.l_spectrogram_accuracy.setObjectName("l_spectrogram_accuracy")
        self.l_word = QtWidgets.QLabel(parent=self.tab_2)
        self.l_word.setGeometry(QtCore.QRect(450, 100, 191, 20))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setUnderline(True)
        self.l_word.setFont(font)
        self.l_word.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.l_word.setObjectName("l_word")
        self.b_record = QtWidgets.QPushButton(parent=self.tab_2)
        self.b_record.setEnabled(False)
        self.b_record.setGeometry(QtCore.QRect(430, 130, 111, 32))
        self.b_record.setObjectName("b_record")
        self.label2 = QtWidgets.QLabel(parent=self.tab_2)
        self.label2.setGeometry(QtCore.QRect(480, 80, 131, 20))
        self.label2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label2.setObjectName("label2")
        self.b_wordbank = QtWidgets.QPushButton(parent=self.tab_2)
        self.b_wordbank.setGeometry(QtCore.QRect(80, 150, 141, 31))
        self.b_wordbank.setAutoFillBackground(False)
        self.b_wordbank.setObjectName("b_wordbank")
        self.label2_3 = QtWidgets.QLabel(parent=self.tab_2)
        self.label2_3.setGeometry(QtCore.QRect(400, 219, 291, 31))
        self.label2_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label2_3.setObjectName("label2_3")
        self.fp_wordbank = QtWidgets.QLineEdit(parent=self.tab_2)
        self.fp_wordbank.setGeometry(QtCore.QRect(80, 120, 221, 21))
        self.fp_wordbank.setObjectName("fp_wordbank")
        self.l_spectrogram_word = QtWidgets.QLabel(parent=self.tab_2)
        self.l_spectrogram_word.setGeometry(QtCore.QRect(450, 250, 191, 20))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setUnderline(True)
        self.l_spectrogram_word.setFont(font)
        self.l_spectrogram_word.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.l_spectrogram_word.setObjectName("l_spectrogram_word")
        self.l_linear_word = QtWidgets.QLabel(parent=self.tab_2)
        self.l_linear_word.setGeometry(QtCore.QRect(450, 331, 191, 20))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setUnderline(True)
        self.l_linear_word.setFont(font)
        self.l_linear_word.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.l_linear_word.setObjectName("l_linear_word")
        self.l_linear_accuracy = QtWidgets.QLabel(parent=self.tab_2)
        self.l_linear_accuracy.setGeometry(QtCore.QRect(460, 351, 171, 20))
        self.l_linear_accuracy.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.l_linear_accuracy.setObjectName("l_linear_accuracy")
        self.label2_6 = QtWidgets.QLabel(parent=self.tab_2)
        self.label2_6.setGeometry(QtCore.QRect(400, 300, 291, 31))
        self.label2_6.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label2_6.setObjectName("label2_6")
        self.b_predict = QtWidgets.QPushButton(parent=self.tab_2)
        self.b_predict.setEnabled(False)
        self.b_predict.setGeometry(QtCore.QRect(540, 130, 111, 32))
        self.b_predict.setObjectName("b_predict")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(parent=MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolBar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.b_record, self.fp_wordbank)
        MainWindow.setTabOrder(self.fp_wordbank, self.b_wordbank)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Test Training"))
        self.l_loopCounter.setWhatsThis(_translate("MainWindow", "word"))
        self.label2_4.setWhatsThis(_translate("MainWindow", "word"))
        self.label2_4.setText(_translate("MainWindow", "Word:"))
        self.b_uploadrecordings.setText(_translate("MainWindow", "Upload recordings"))
        self.label4_2.setWhatsThis(_translate("MainWindow", "word"))
        self.label4_2.setText(_translate("MainWindow", "Word bank CSV or TXT file path:"))
        self.l_word_2.setWhatsThis(_translate("MainWindow", "word"))
        self.l_word_2.setText(_translate("MainWindow", "example word"))
        self.b_record_2.setText(_translate("MainWindow", "Record"))
        self.label2_5.setWhatsThis(_translate("MainWindow", "word"))
        self.label2_5.setText(_translate("MainWindow", "Current loop of word bank:"))
        self.label3.setWhatsThis(_translate("MainWindow", "word"))
        self.label3.setText(_translate("MainWindow", "File path of pre-recorded vocab:"))
        self.b_process.setText(_translate("MainWindow", "TRAIN MODEL with CURRENT RECORDINGS"))
        self.label5.setWhatsThis(_translate("MainWindow", "word"))
        self.label5.setText(_translate("MainWindow", "Current file path that processor looks at:"))
        self.fp_1process.setPlaceholderText(_translate("MainWindow", "filepath/of/currentrecordings"))
        self.b_wordbank_2.setText(_translate("MainWindow", "Upload word bank"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Record Vocab"))
        self.b_generatenew.setText(_translate("MainWindow", "Generate new word"))
        self.label2_2.setWhatsThis(_translate("MainWindow", "word"))
        self.label2_2.setText(_translate("MainWindow", "Available words:"))
        self.label4.setWhatsThis(_translate("MainWindow", "word"))
        self.label4.setText(_translate("MainWindow", "Word bank CSV or TXT file path:"))
        self.l_spectrogram_accuracy.setWhatsThis(_translate("MainWindow", "word"))
        self.l_spectrogram_accuracy.setText(_translate("MainWindow", "0% accurate"))
        self.l_word.setWhatsThis(_translate("MainWindow", "word"))
        self.l_word.setText(_translate("MainWindow", "example word"))
        self.b_record.setText(_translate("MainWindow", "Record"))
        self.label2.setWhatsThis(_translate("MainWindow", "word"))
        self.label2.setText(_translate("MainWindow", "Word to test model:"))
        self.b_wordbank.setText(_translate("MainWindow", "Upload word bank"))
        self.label2_3.setWhatsThis(_translate("MainWindow", "word"))
        self.label2_3.setText(_translate("MainWindow", "Word prediction, spectrogram method:"))
        self.fp_wordbank.setPlaceholderText(_translate("MainWindow", "default file set in Python code"))
        self.l_spectrogram_word.setWhatsThis(_translate("MainWindow", "word"))
        self.l_spectrogram_word.setText(_translate("MainWindow", "example word"))
        self.l_linear_word.setWhatsThis(_translate("MainWindow", "word"))
        self.l_linear_word.setText(_translate("MainWindow", "example word"))
        self.l_linear_accuracy.setWhatsThis(_translate("MainWindow", "word"))
        self.l_linear_accuracy.setText(_translate("MainWindow", "0% accurate"))
        self.label2_6.setWhatsThis(_translate("MainWindow", "word"))
        self.label2_6.setText(_translate("MainWindow", "Word prediction, linear prediction method:"))
        self.b_predict.setText(_translate("MainWindow", "Predict"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Recognition"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
