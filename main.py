import sys
import csv
import time
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout,
    QWidget, QPushButton, QHBoxLayout, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
import pyqtgraph as pg
from scipy.signal import find_peaks, butter, filtfilt
import threading
from datetime import datetime
import os
from sensor.max30102 import MAX30102

IR_THRESHOLD = 30000

class DataEmitter(QObject):
    new_data = pyqtSignal(list)


class PulsePlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.setTitle("MAX30102 Pulse Monitor", color='k', size='16pt')
        self.setLabel('left', "IR Value", **{'color': 'black', 'font-size': '12pt'})
        self.setLabel('bottom', "Sample", **{'color': 'black', 'font-size': '12pt'})
        self.showGrid(x=True, y=True)
        self.curve = self.plot(pen=pg.mkPen(color='r', width=2))
        self.buffer_size = 300
        self.data = np.zeros(self.buffer_size, dtype=int)
        self.ptr = 0

    def update_plot(self, new_vals):
        n = len(new_vals)
        if n > self.buffer_size:
            new_vals = new_vals[-self.buffer_size:]
            n = self.buffer_size

        if self.ptr + n < self.buffer_size:
            self.data[self.ptr:self.ptr + n] = new_vals
        else:
            till_end = self.buffer_size - self.ptr
            self.data[self.ptr:] = new_vals[:till_end]
            self.data[:n - till_end] = new_vals[till_end:]
        self.ptr = (self.ptr + n) % self.buffer_size

        if self.ptr == 0:
            display_data = self.data
        else:
            display_data = np.concatenate((self.data[self.ptr:], self.data[:self.ptr]))

        self.curve.setData(display_data)


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAX30102 Pulse Monitor")
        self.resize(700, 500)

        self.plot = PulsePlot(self)
        self.label_bpm = QLabel("BPM: --", alignment=Qt.AlignCenter)
        self.label_bpm.setStyleSheet("font-size: 24pt; font-weight: bold; color: #333;")

        self.label_status = QLabel("Status: Waiting...", alignment=Qt.AlignCenter)
        self.label_status.setStyleSheet("font-size: 14pt; color: #666;")

        self.btn_toggle = QPushButton("Start Measurement")
        self.btn_toggle.setStyleSheet("padding: 10px; font-size: 12pt;")
        self.btn_toggle.clicked.connect(self.toggle_measurement)

        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setStyleSheet("padding: 10px; font-size: 12pt;")
        self.btn_export.clicked.connect(self.export_csv)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_toggle)
        button_layout.addWidget(self.btn_export)

        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        layout.addWidget(self.label_bpm)
        layout.addWidget(self.label_status)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.sensor = MAX30102()
        # self.sensor.setup(led_current=0x1F, sample_rate=100, pulse_width=411)  # Optional config

        self.ir_data = []
        self.recent_ir_values = []
        self.csv_data = []
        self.measuring = False

        self.data_emitter = DataEmitter()
        self.data_emitter.new_data.connect(self.update_plot)

        self.thread = threading.Thread(target=self.read_sensor)
        self.thread.daemon = True
        self.thread.start()

    def toggle_measurement(self):
        self.measuring = not self.measuring
        if self.measuring:
            self.ir_data.clear()
            self.csv_data.clear()
            self.label_bpm.setText("BPM: --")
            self.label_status.setText("Status: Waiting for finger...")
            self.btn_toggle.setText("Stop Measurement")
        else:
            self.btn_toggle.setText("Start Measurement")
            self.export_csv(auto=True)

    def read_sensor(self):
        batch_size = 10
        batch = []

        while True:
            try:
                red, ir = self.sensor.read_fifo()

                if ir < 5000 or ir > 150000:
                    continue

                self.recent_ir_values.append(ir)
                if len(self.recent_ir_values) > 50:
                    self.recent_ir_values.pop(0)

                avg_ir = np.mean(self.recent_ir_values)
                finger_detected = ir > avg_ir * 1.05

                batch.append(ir)
                if len(batch) >= batch_size:
                    self.data_emitter.new_data.emit(batch)
                    batch.clear()

                if self.measuring:
                    timestamp = time.time()
                    readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

                    if finger_detected:
                        self.ir_data.append(ir)
                        if len(self.ir_data) > 300:
                            self.ir_data = self.ir_data[-300:]

                        bpm = 0
                        status = "No data"

                        if len(self.ir_data) >= 200:
                            bpm = self.calc_bpm(self.ir_data)
                            status = self.classify_bpm(bpm)
                            self.label_bpm.setText(f"BPM: {bpm:.1f}")
                            self.label_status.setText(f"Status: {status}")
                    else:
                        self.ir_data.clear()
                        bpm = 0
                        status = "Waiting for finger..."
                        self.label_bpm.setText("BPM: --")
                        self.label_status.setText(f"Status: {status}")

                    self.csv_data.append((timestamp, readable_time, ir, f"{bpm:.1f}", status))

                else:
                    self.label_bpm.setText("BPM: --")
                    self.label_status.setText("Status: Finger off")

                time.sleep(0.01)
            except Exception as e:
                print(f"Error reading sensor: {e}")
                time.sleep(1)

    def update_plot(self, new_vals):
        self.plot.update_plot(new_vals)

    def moving_average(self, data, window_size=5):
        if len(data) < window_size:
            return np.array(data)
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def butter_lowpass_filter(self, data, cutoff=2.5, fs=100, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def calc_bpm(self, ir_data, fs=100):
        smoothed = self.moving_average(ir_data, window_size=5)
        filtered = self.butter_lowpass_filter(smoothed, cutoff=2.5, fs=fs, order=2)
        peaks, _ = find_peaks(filtered, distance=fs * 0.7, prominence=0.2 * np.max(filtered))
        if len(peaks) < 2:
            return 0
        rr_intervals = np.diff(peaks) / fs
        bpm = 60 / np.mean(rr_intervals)
        return bpm

    def classify_bpm(self, bpm):
        if bpm < 60:
            return "Bradycardia"
        elif bpm > 100:
            return "Tachycardia"
        else:
            return "Normal"

    def export_csv(self, auto=False):
        if not self.csv_data:
            return

        if auto:
            folder = os.path.expanduser('~/max30102_exports')
            os.makedirs(folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = os.path.join(folder, f"max30102_data_{timestamp}.csv")
        else:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
            if not file_name:
                return

        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Readable Time', 'IR_Value', 'BPM', 'Status'])
            writer.writerows(self.csv_data)

        print("Data exported to:", file_name)

    def closeEvent(self, event):
        self.sensor.shutdown()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()