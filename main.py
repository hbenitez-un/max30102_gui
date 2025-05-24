import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from scipy.signal import find_peaks, butter, filtfilt
import pyqtgraph as pg
import threading
import time

# Sensor
from sensor.max30102 import MAX30102


def butter_lowpass_filter(data, cutoff=2.5, fs=100, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAX30102 Heart Rate Monitor")
        self.resize(600, 400)

        # === GUI Elements ===
        self.label_bpm = QLabel("BPM: --")
        self.label_bpm.setStyleSheet("font-size: 40px; font-weight: bold; color: red;")
        self.label_status = QLabel("Status: --")
        self.label_status.setStyleSheet("font-size: 20px;")

        self.graph = pg.PlotWidget()
        self.graph.setYRange(0, 200000)
        self.graph.setTitle("IR Signal")
        self.plot_line = self.graph.plot(pen='r')

        layout = QVBoxLayout()
        layout.addWidget(self.graph)
        layout.addWidget(self.label_bpm)
        layout.addWidget(self.label_status)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # === Sensor + Data ===
        self.sensor = MAX30102()
        self.ir_data = []
        self.last_bpm_time = time.time()

        # Start sensor reading thread
        self.thread = threading.Thread(target=self.read_sensor)
        self.thread.daemon = True
        self.thread.start()

        # Update graph with timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)  # 10 FPS aprox

    def read_sensor(self):
        while True:
            try:
                red, ir = self.sensor.read_fifo()
                self.ir_data.append(ir)

                # Mantén los últimos 300 puntos (3s)
                if len(self.ir_data) > 300:
                    self.ir_data = self.ir_data[-300:]

                # Calcular BPM cada 1s si hay suficientes muestras
                if len(self.ir_data) >= 200 and time.time() - self.last_bpm_time > 1:
                    bpm = self.calc_bpm(self.ir_data)
                    self.label_bpm.setText(f"{bpm:.1f} BPM")
                    self.label_status.setText(self.classify_bpm(bpm))
                    self.last_bpm_time = time.time()

                time.sleep(0.05)  # 20 Hz
            except Exception as e:
                print(f"Sensor error: {e}")
                time.sleep(1)

    def update_plot(self):
        if self.ir_data:
            self.plot_line.setData(self.ir_data)

    def calc_bpm(self, data, fs=100):
        try:
            filtered = butter_lowpass_filter(data, cutoff=2.5, fs=fs, order=2)
            peaks, _ = find_peaks(filtered, distance=fs * 0.6)
            if len(peaks) < 2:
                return 0
            rr = np.diff(peaks) / fs
            return 60 / np.mean(rr)
        except Exception as e:
            print(f"BPM error: {e}")
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
