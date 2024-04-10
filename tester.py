import sys
from PyQt6 import QtWidgets, uic
from MainWindow import Ui_MainWindow
import sounddevice as sd
import queue
import os
from datetime import datetime
from time import gmtime, strftime
import random




# /Users/mir/Documents/GITHUB/vocab-recognition/vocabList.csv

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.b_process.clicked.connect(self.onProcessBtnClicked)
        self.b_wordbank.clicked.connect(self.onWordBankBtnClicked)
        self.b_uploadrecordings.clicked.connect(self.onUploadRecordingsBtnClicked)
        self.b_record.clicked.connect(self.onRecordBtnClicked)

        self.wordBankFilled = False
        self.wordBankIndex = 0
        self.wordBankContent = 0

    def onProcessBtnClicked(self):
        print("clicked the process button!")

    def onWordBankBtnClicked(self):
        print("clicked the word bank button!")
        wordbankFP = str(self.fp_wordbank.text())
        print(wordbankFP)
        try:
            with open(wordbankFP) as file:
                self.wordBankContent = file.readlines()
            # header = content[:1]
            # rows = content[1:]
            print(self.wordBankContent)

            print(self.wordBankContent[1])
            if not self.wordBankFilled:
                for word in self.wordBankContent:
                    self.plainTextEdit.insertPlainText(str(word))
                self.wordBankFilled = True
            self.handleWordBank()
        except:
            print("file doesnt exist or other error in wordbankbtnclicked")


    def onUploadRecordingsBtnClicked(self):
        print("clicked upload recordings button!")

    def handleWordBank(self):
        print("word bank is being handled!")
        if self.wordBankFilled:
            wordBankLen = len(self.wordBankContent)
            self.randomWordIndeces = random.sample(range(wordBankLen), k=wordBankLen)
            print(self.randomWordIndeces)
            
            self.wordBankIndex = self.randomWordIndeces[1]
            insideIdx = 0
            for word in self.wordBankContent:

                if insideIdx == self.wordBankIndex:
                # print(self.wordBankIndex)
                    self.l_word.setText(word)
                    break
                insideIdx = insideIdx + 1
                # print
            # print(self.wordBankContent(self.wordBankIndex))




    def onRecordBtnClicked(self):
        print("clicked record button!")
        # time to record !
        fs = 44100
        filename = str(self.fp_1process.text())

        try:
            if len(filename)<3:
                filename = os.getcwd()
                filename = filename + "/recording_"
                dt = strftime('%d-%b-%H%M%S')
                # dt.strftime('%m%d-%H%M')
                filename = filename + dt
                filename = filename + ".mp3"

            import sounddevice as sd
            import soundfile as sf
            import numpy  # Make sure NumPy is loaded before it is used in the callback
            assert numpy  # avoid "imported but unused" message (W0611)

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
                    while True:
                        file.write(q.get())


            # NOW CHANGE THE VISIBLE WORD IN THE WORD BANK!

        except KeyboardInterrupt:
            print('\nRecording finished: ' + repr(filename))
            # parser.exit(0)
        except Exception as e:
            print("error:")
            print(e)
            # parser.exit(type(e).__name__ + ': ' + str(e))



    


app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()
