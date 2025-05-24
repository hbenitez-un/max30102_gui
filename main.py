import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import threading
import time
from scipy.signal import find_peaks, butter, filtfilt

# Importa el driver del sensor
from sensor.max30102 import MAX30102


class PulsePlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.setTitle("MAX30102 Pulse Monitor", color='k', size='16pt')
        self.setLabel('left', "IR Value", **{'color': 'black', 'font-size': '12pt'})
        self.setLabel('bottom', "Sample", **{'color': 'black', 'font-size': '12pt'})
        self.showGrid(x=True, y=True)
        self.curve = self.plot(pen=pg.mkPen(color='r', width=2))
        self.data = [0] * 300

    def update_plot(self, new_val):
        self.data = self.data[1:] + [new_val]
        self.curve.setData(self.data)


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAX30102 Pulse Monitor")
        self.resize(600, 400)

        self.plot = PulsePlot(self)

        self.label_bpm = QLabel("BPM: 0")
        self.label_bpm.setAlignment(Qt.AlignCenter)
        self.label_bpm.setStyleSheet("font-size: 24pt; font-weight: bold; color: #333;")

        self.label_status = QLabel("Status: Normal")
        self.label_status.setAlignment(Qt.AlignCenter)
        self.label_status.setStyleSheet("font-size: 14pt; color: #666;")

        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        layout.addWidget(self.label_bpm)
        layout.addWidget(self.label_status)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.sensor = MAX30102()
        self.thread = threading.Thread(target=self.read_sensor)
        self.thread.daemon = True
        self.thread.start()

    def read_sensor(self):
        ir_data = []
        while True:
            try:
                red, ir = self.sensor.read_fifo()
                ir_data.append(ir)
                self.plot.update_plot(ir)

                if len(ir_data) > 300:
                    ir_data = ir_data[-300:]

                if len(ir_data) >= 200:
                    bpm = self.calc_bpm(ir_data)
                    self.label_bpm.setText(f"BPM: {bpm:.1f}")
                    self.label_status.setText(f"Status: {self.classify_bpm(bpm)}")

                time.sleep(0.05)
            except Exception as e:
                print(f"Error reading sensor: {e}")
                time.sleep(1)

    def butter_lowpass_filter(self, data, cutoff=2.5, fs=100, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def calc_bpm(self, ir_data, fs=100):
        filtered = self.butter_lowpass_filter(ir_data, cutoff=2.5, fs=fs, order=2)
        peaks, _ = find_peaks(filtered, distance=fs * 0.6)
        if len(peaks) < 2:
            return 0
        rr = np.diff(peaks) / fs
        return 60 / np.mean(rr)

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
