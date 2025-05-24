import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import threading
import time
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from PyQt5.QtCore import QTimer
from scipy.signal import find_peaks, butter, filtfilt
import threading
import time

# Import the sensor driver
from sensor.max30102 import MAX30102


def butter_lowpass_filter(data, cutoff=2.5, fs=100, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Rate Monitor")
        self.resize(300, 200)

        # Minimal BPM display
        self.label_bpm = QLabel("BPM: --")
        self.label_bpm.setStyleSheet("font-size: 48px; font-weight: bold; color: red;")
        self.label_status = QLabel("Status: --")
        self.label_status.setStyleSheet("font-size: 20px;")

        layout = QVBoxLayout()
        layout.addWidget(self.label_bpm)
        layout.addWidget(self.label_status)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.sensor = MAX30102()
        self.ir_data = []
        self.last_bpm_time = time.time()

        self.thread = threading.Thread(target=self.read_sensor)
        self.thread.daemon = True
        self.thread.start()

    def read_sensor(self):
        while True:
            try:
                red, ir = self.sensor.read_fifo()
                self.ir_data.append(ir)

                # Mantener buffer limitado a 300 muestras (~3 segundos a 100 Hz)
                if len(self.ir_data) > 300:
                    self.ir_data = self.ir_data[-300:]

                # Calcular BPM cada 1 segundo si hay suficientes datos
                if len(self.ir_data) >= 200 and time.time() - self.last_bpm_time > 1:
                    bpm = self.calc_bpm(self.ir_data)
                    self.label_bpm.setText(f"{bpm:.1f} BPM")
                    self.label_status.setText(self.classify_bpm(bpm))
                    self.last_bpm_time = time.time()

                time.sleep(0.05)  # 20 Hz
            except Exception as e:
                print(f"Error reading sensor: {e}")
                time.sleep(1)

    def calc_bpm(self, data, fs=100):
        try:
            filtered = butter_lowpass_filter(data, cutoff=2.5, fs=fs, order=2)
            peaks, _ = find_peaks(filtered, distance=fs * 0.6)
            if len(peaks) < 2:
                return 0
            rr_intervals = np.diff(peaks) / fs
            bpm = 60 / np.mean(rr_intervals)
            return bpm
        except Exception as e:
            print(f"Error in BPM calc: {e}")
            return 0

    def classify_bpm(self, bpm):
        if bpm < 60:
            return "Bradycardia"
        elif bpm > 100:
            return "Tachycardia"
        return "Normal"

    def closeEvent(self, event):
        self.sensor.shutdown()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
