import sys
from PyQt6 import QtWidgets, uic
from MainWindow import Ui_MainWindow
import sounddevice as sd
import queue
import os
from datetime import datetime
from time import gmtime, strftime
import random
import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)
import threading


# /Users/mir/Documents/GITHUB/vocab-recognition/vocabList.csv

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.b_process.clicked.connect(self.onProcessBtnClicked)
        self.b_wordbank.clicked.connect(self.onWordBankBtnClicked)
        self.b_uploadrecordings.clicked.connect(self.onUploadRecordingsBtnClicked)
        self.b_record.clicked.connect(self.onRecStopBtnClicked)

        self.wordBankFilled = False
        self.wordBankIndex = 0
        self.wordBankContent = 0
        self.bool_recording = False
        self.curWordIndex = 0
        self.curWord = 'default'
        self.wordBankLen = 1

    def onProcessBtnClicked(self):
        print("clicked the process button!")

    def onWordBankBtnClicked(self):
        print("clicked the word bank button!")
        wordbankFP = str(self.fp_wordbank.text())
        if len(wordbankFP) < 1:
            wordbankFP = "/Users/mir/Documents/GITHUB/vocab-recognition/vocabList.csv"
        print(wordbankFP)
        try:
            with open(wordbankFP) as file:
                self.wordBankContent = [line.strip() for line in file.readlines()] 

            # header = content[:1]
            # rows = content[1:]
            print(self.wordBankContent)

            print(self.wordBankContent[1])
            if not self.wordBankFilled:
                for word in self.wordBankContent:
                    self.plainTextEdit.insertPlainText(str(word))
                    self.plainTextEdit.insertPlainText(str("\n"))
                self.wordBankFilled = True
                self.handleWordBank()
        except:
            print("file doesnt exist or other error in wordbankbtnclicked")


    def onUploadRecordingsBtnClicked(self):
        print("clicked upload recordings button!")

    def handleWordBank(self):
        print("word bank is being handled!")
        self.b_record.setEnabled(True)
        if self.wordBankFilled:
            self.wordBankLen = len(self.wordBankContent)
            self.randomWordIndeces = random.sample(range(self.wordBankLen), k=self.wordBankLen)
            print(self.randomWordIndeces)
            print("content:", self.wordBankContent)
            print(self.wordBankContent[1])
            
            self.curWordIndex = 0
            self.curWord = self.wordBankContent[self.randomWordIndeces[self.curWordIndex]]
            self.l_word.setText(self.curWord)
        
    def setNewWord(self):
        self.curWordIndex = self.curWordIndex + 1
        if self.curWordIndex > self.wordBankLen:
            self.handleWordBank()
        else:
            self.curWord = self.wordBankContent[self.randomWordIndeces[self.curWordIndex]]
            self.l_word.setText(self.curWord)

    def start_recording(self):
        fs = 44100
        filename = str(self.fp_1process.text())

        try:
            if len(filename) < 3:
                filename = os.getcwd()
                filename = filename + "/" + self.curWord + "/"
                if not os.path.exists(filename):
                    os.makedirs(filename)
                filename = filename + "/" + self.curWord + "_"
                dt = strftime('%d-%b-%H%M%S')
                filename = filename + dt
                filename = filename + ".mp3"

            q = queue.Queue()

            def callback(indata, frames, time, status):
                """This is called (from a separate thread) for each audio block."""
                if status:
                    print(status, file=sys.stderr)
                q.put(indata.copy())

            # Make sure the file is opened before recording anything:
            with sf.SoundFile(filename, mode='x', samplerate=fs,
                            channels=1) as file:
                with sd.InputStream(samplerate=fs,
                                    channels=1, callback=callback):
                    print('#' * 80)
                    print('press Ctrl+C to stop the recording')
                    print('#' * 80)
                    # self.b_record.setText("Click to stop recording")
                    while not self.stop_recording:
                        file.write(q.get())

            # NOW CHANGE THE VISIBLE WORD IN THE WORD BANK!

        except Exception as e:
            print("error:")
            print(e)

    def onRecordBtnClicked(self):
        print("clicked record/stop button!")
        # time to record!
        self.stop_recording = False
        threading.Thread(target=self.start_recording).start()

    def onStopBtnClicked(self):
        print("clicked stop button!")
        self.stop_recording = True
        self.setNewWord()


    def onRecStopBtnClicked(self):
        if not self.bool_recording:
            self.onRecordBtnClicked()
            self.b_record.setText("STOP recording")
            self.bool_recording = True

        else:
            self.onStopBtnClicked()
            self.b_record.setText("Record")
            self.bool_recording = False




app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()
