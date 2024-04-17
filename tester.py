import sys
from PyQt6 import QtWidgets, uic
# from MainWindow import Ui_MainWindow
from ui_testtraining import Ui_MainWindow
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
from scipy.io import wavfile
import numpy as np

# /Users/mir/Documents/GITHUB/vocab-recognition/vocabList.csv
# /Users/mir/Documents/GITHUB/vocab-recognition/vocabList2.csv
# baseFolder = "/Users/mir/Documents/GITHUB/vocab-recognition/DATASET_48k"


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.b_process.clicked.connect(self.onProcessBtnClicked)
        self.b_wordbank.clicked.connect(self.onWordBankBtnClicked)
        self.b_wordbank_2.clicked.connect(self.onWordBank2BtnClicked)
        self.b_uploadrecordings.clicked.connect(self.onUploadRecordingsBtnClicked)
        self.b_record.clicked.connect(self.onRecStopBtnClicked)
        self.b_record_2.clicked.connect(self.onRecStopBtn2Clicked)
        self.b_predict.clicked.connect(self.onPredictBtnClicked)
        self.b_generatenew.clicked.connect(self.onGenerateNewClicked)


        self.b_predict.setEnabled(False)

        self.wordBankFilled = False
        self.wordBank2Filled = False
        self.wordBankIndex = 0
        self.wordBankContent = 0
        self.bool_recording = False
        self.curWordIndex = 0
        self.curWord = 'default'
        self.wordBankLen = 1
        self.filename = ""
        self.loopCounter = -1

        # make a new folder with time and date for all the files we will record with this instance of the GUI!
        dt = strftime('%d-%b-%H%M%S')
        self.curfolderpath = os.getcwd() + "/" + dt
        if not os.path.exists(self.curfolderpath):
            os.makedirs(self.curfolderpath)


    def onProcessBtnClicked(self):
        print("clicked the process button! do training of the model here")
        self.b_process.setText("training model, go to recognition tab next")
        # self.b_process.setEnabled(False)
        # self.tabWidget.setCurrentIndex(1)



    def onWordBankBtnClicked(self):
        print("clicked the word bank button!")
        wordbankFP = str(self.fp_wordbank.text())
        if len(wordbankFP) < 1:
            wordbankFP = "/Users/mir/Documents/GITHUB/vocab-recognition/vocabList2.csv"
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
                self.handleWordBank(0)
        except:
            print("file doesnt exist or other error in wordbankbtnclicked")

    def onWordBank2BtnClicked(self):
        print("clicked the word bank 2 button!")
        wordbank2FP = str(self.fp_wordbank_2.text())
        if len(wordbank2FP) < 1:
            wordbank2FP = "/Users/mir/Documents/GITHUB/vocab-recognition/vocabList2.csv"
        print(wordbank2FP)
        try:
            with open(wordbank2FP) as file:
                self.wordBank2Content = [line.strip() for line in file.readlines()] 

            # header = content[:1]
            # rows = content[1:]
            print(self.wordBank2Content)

            print(self.wordBank2Content[1])
            if not self.wordBank2Filled:
                for word in self.wordBank2Content:
                    self.plainTextEdit_2.insertPlainText(str(word))
                    self.plainTextEdit_2.insertPlainText(str("\n"))
                self.wordBank2Filled = True
                self.handleWordBank(1)
        except:
            print("file doesnt exist or other error in wordbankbtnclicked")

    def onUploadRecordingsBtnClicked(self):
        print("clicked upload recordings button!")

    def handleWordBank(self,whichWordBank):
        print("word bank is being handled!")
        if whichWordBank == 1:
            self.loopCounter = self.loopCounter + 1
            self.b_record_2.setEnabled(True)
        if whichWordBank == 0:
            self.b_record.setEnabled(True)
        if self.wordBankFilled and whichWordBank==0:
            self.wordBankLen = len(self.wordBankContent)
            self.randomWordIndeces = random.sample(range(self.wordBankLen), k=self.wordBankLen)
            print(self.randomWordIndeces)
            print("content:", self.wordBankContent)
            print(self.wordBankContent[1])
            self.curWordIndex = 0
            self.curWord = self.wordBankContent[self.randomWordIndeces[self.curWordIndex]]
            self.l_word.setText(self.curWord)
        if self.wordBank2Filled and whichWordBank ==1:
            self.wordBank2Len = len(self.wordBank2Content)
            self.randomWord2Indeces = random.sample(range(self.wordBank2Len), k=self.wordBank2Len)
            print(self.randomWord2Indeces)
            print("content:", self.wordBank2Content)
            print(self.wordBank2Content[1])
            self.curWord2Index = 0
            self.curWord2 = self.wordBank2Content[self.randomWord2Indeces[self.curWord2Index]]
            self.l_word_2.setText(self.curWord2)
            self.l_loopCounter.setText(str(self.loopCounter) + ", " + str(self.curWordIndex) + "/" + str(self.wordBankLen))
        
        
    def setNewWord(self):
        self.curWordIndex = self.curWordIndex + 1
        if self.curWordIndex > (self.wordBankLen - 1):
            self.handleWordBank(0)
        else:
            self.curWord = self.wordBankContent[self.randomWordIndeces[self.curWordIndex]]
            self.l_word.setText(self.curWord)

    def setNewWord2(self):
        self.curWord2Index = self.curWord2Index + 1
        if self.curWord2Index > (self.wordBank2Len - 1):
            self.handleWordBank(1)
        else:
            self.curWord2 = self.wordBankContent[self.randomWordIndeces[self.curWord2Index]]
            self.l_word_2.setText(self.curWord2)
            self.l_loopCounter.setText(str(self.loopCounter) + ", " + str(self.curWord2Index) + "/" + str(self.wordBank2Len))

    def onGenerateNewClicked(self):
        self.setNewWord2()

    def start_recording(self):
        fs = 48000
        filename = str(self.fp_1process.text())
        try:
            if len(filename) < 3:
                # filename = os.getcwd()
                filename = self.curfolderpath
                filename = filename + "/" + "recognition" + "/" + self.curWord + "/"
                if not os.path.exists(filename):
                    os.makedirs(filename)
                filename = filename + "/" + self.curWord + "_"
                # dt = strftime('%d-%b-%H%M%S')
                filename = filename + str(self.loopCounter)
                filename = filename + ".wav"
                self.filename = filename
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

        except Exception as e:
            print("error:")
            print(e)

    def start_recording2(self):
        fs = 48000
        filename = str(self.fp_1process.text())
        try:
            if len(filename) < 3:
                # filename = os.getcwd()
                filename = self.curfolderpath
                filename = filename + "/" + "training" + "/" + self.curWord2 + "/"
                if not os.path.exists(filename):
                    os.makedirs(filename)
                filename = filename + "/" + self.curWord2 + "_"
                # dt = strftime('%d-%b-%H%M%S')
                filename = filename + str(self.loopCounter)
                filename = filename + ".wav"
                self.filename = filename
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

        except Exception as e:
            print("error:")
            print(e)



    def onRecordBtnClicked(self):
        print("clicked record/stop button!")
        # time to record!
        self.stop_recording = False
        threading.Thread(target=self.start_recording).start()
    def onRecordBtn2Clicked(self):
        print("clicked record/stop button!")
        # time to record!
        self.stop_recording = False
        threading.Thread(target=self.start_recording2).start()

    def onStopBtnClicked(self):
        print("clicked stop button!")
        self.stop_recording = True
        try:
            cut_files(self.filename)
        except:
            print("error in cut files")
        # dont set new word, now you have to click predict and it will predict
        # self.setNewWord()
    
    def onStopBtn2Clicked(self):
        print("clicked stop button!")
        self.stop_recording = True
        try:
            cut_files(self.filename)
        except:
            print("error in cut files")
        self.setNewWord2()

    def onRecStopBtnClicked(self):
        if not self.bool_recording:
            self.onRecordBtnClicked()
            self.b_record.setText("STOP recording")
            self.bool_recording = True

        else:
            self.onStopBtnClicked()
            self.b_record.setText("Record")
            self.bool_recording = False

    def onRecStopBtn2Clicked(self):
        if not self.bool_recording:
            self.onRecordBtn2Clicked()
            self.b_record_2.setText("STOP recording")
            self.bool_recording = True

        else:
            self.onStopBtn2Clicked()
            self.b_record_2.setText("Record")
            self.bool_recording = False

    def onPredictBtnClicked(self):
        self.b_predict.setEnabled(True)
        print("predict button clicked. run the machine learning code now!")



        # self.l_pred



    

def cut_files(filepath):
    window_size = 1500
    hop_size = 512
    onset_threshold = 0.02
    offset_threshold = 0.0002
    # input_file_path = 'wav_file.wav' #change the name to our wav file
    input_file_path = filepath
    # sample_rate, audio = wavfile.read(input_file_path)
    audio,sample_rate = sf.read(input_file_path)

    onsets, offsets = detect_first_onset_offset(audio, sample_rate, window_size, hop_size, onset_threshold, offset_threshold)
    onset_sample = int(onsets * int(sample_rate))
    offset_sample = int(offsets * int(sample_rate))

    audio_segment = audio[onset_sample:offset_sample]
    # output_file_path = 'new_wav_file.wav'
    filename = filepath.rsplit('/',1)[1]
    filepath = filepath.rsplit('/',1)[0]
    newfilepath = filepath + "/shortened"
    if not os.path.exists(newfilepath):
        os.makedirs(newfilepath)
    output_file_path = newfilepath + "/" + filename 
    # wavfile.write(output_file_path, sample_rate, audio_segment)
    sf.write(output_file_path,audio_segment,sample_rate)

def short_time_energy(signal, window_size, hop_size):

    energy = np.array([
        np.sum(np.abs(signal[i:i+window_size])**2)
        for i in range(0, len(signal) - window_size, hop_size)
    ])
    return energy / np.max(energy)  # Normalize energy

# Detect onsets and offsets in the audio signal.
def detect_first_onset_offset(signal, sample_rate, window_size, hop_size, onset_threshold, offset_threshold):
    # Normalize signal
    signal = signal / np.max(np.abs(signal))
    
    # Calculate Short-Time Energy
    ste = short_time_energy(signal, window_size, hop_size)
    
    first_onset = None
    first_offset = None
    is_onset = False
    
    # Detect the first onset and its corresponding offset
    for i in range(1, len(ste)):
        if ste[i] > onset_threshold and ste[i-1] <= onset_threshold and not is_onset:
            first_onset = i * hop_size / sample_rate
            is_onset = True
        elif is_onset and ste[i] < offset_threshold and ste[i-1] >= offset_threshold:
            first_offset = i * hop_size / sample_rate
            break  # Stop after finding the first offset
    
    # If an onset was found but no offset, consider the end of the signal as the offset
    if is_onset and first_offset is None:
        first_offset = len(signal) / sample_rate
    if first_onset is None:
        first_onset = 0
    if first_offset is None:
        first_offset = len(signal) / sample_rate
    return first_onset, first_offset

app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()
