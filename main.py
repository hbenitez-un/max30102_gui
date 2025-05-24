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

IR_THRESHOLD = 30000  # Adjust based on your sensor readings

class DataEmitter(QObject):
    """
    QObject subclass to define signals for inter-thread communication.
    Emits new sensor data in batches to the main GUI thread.
    """
    new_data = pyqtSignal(list)  # Signal emits a list of new IR values


class PulsePlot(pg.PlotWidget):
    """
    Custom PyQtGraph widget for plotting IR values from MAX30102 sensor.
    Uses a circular buffer to minimize latency and improve plotting performance.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.setTitle("MAX30102 Pulse Monitor", color='k', size='16pt')
        self.setLabel('left', "IR Value", **{'color': 'black', 'font-size': '12pt'})
        self.setLabel('bottom', "Sample", **{'color': 'black', 'font-size': '12pt'})
        self.showGrid(x=True, y=True)

        # Initialize plotting curve
        self.curve = self.plot(pen=pg.mkPen(color='r', width=2))

        # Buffer to hold IR data samples (fixed size)
        self.buffer_size = 300
        self.data = np.zeros(self.buffer_size, dtype=int)
        self.ptr = 0  # Circular buffer pointer

    def update_plot(self, new_vals):
        """
        Update the plot with new IR values.
        Insert new_vals into the circular buffer, then update the graph.
        Args:
            new_vals (list or array): new IR sensor data samples to add
        """
        n = len(new_vals)
        if n > self.buffer_size:
            # If batch too large, keep only most recent samples
            new_vals = new_vals[-self.buffer_size:]
            n = self.buffer_size

        # Insert new values into circular buffer
        if self.ptr + n < self.buffer_size:
            self.data[self.ptr:self.ptr + n] = new_vals
        else:
            till_end = self.buffer_size - self.ptr
            self.data[self.ptr:] = new_vals[:till_end]
            self.data[:n - till_end] = new_vals[till_end:]
        self.ptr = (self.ptr + n) % self.buffer_size

        # Rearrange buffer to plot in correct order
        if self.ptr == 0:
            display_data = self.data
        else:
            display_data = np.concatenate((self.data[self.ptr:], self.data[:self.ptr]))

        self.curve.setData(display_data)


class MainApp(QMainWindow):
    """
    Main application window.
    Controls sensor reading, plotting, BPM calculation,
    heart rate classification, and CSV data export.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAX30102 Pulse Monitor")
        self.resize(700, 500)

        # Setup UI components
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

        # Layout setup for buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_toggle)
        button_layout.addWidget(self.btn_export)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        layout.addWidget(self.label_bpm)
        layout.addWidget(self.label_status)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initialize sensor interface and data containers
        self.sensor = MAX30102()
        self.ir_data = []       # Recent IR data for BPM calculation
        self.csv_data = []      # Data for CSV export (timestamp, readable time, IR, BPM, status)
        self.measuring = False  # Measurement state flag

        # Setup signal emitter for thread-safe data update
        self.data_emitter = DataEmitter()
        self.data_emitter.new_data.connect(self.update_plot)

        # Start sensor reading thread (daemon thread)
        self.thread = threading.Thread(target=self.read_sensor)
        self.thread.daemon = True
        self.thread.start()

        # Threshold to detect if finger is on the sensor (adjust experimentally)
        self.finger_on_threshold = 5000

    def toggle_measurement(self):
        """
        Start or stop the measurement.
        Clears data when starting.
        Auto exports CSV when stopping.
        """
        self.measuring = not self.measuring
        if self.measuring:
            self.ir_data.clear()
            self.csv_data.clear()
            self.label_bpm.setText("BPM: --")
            self.label_status.setText("Status: Waiting for finger...")
            self.btn_toggle.setText("Stop Measurement")
        else:
            self.btn_toggle.setText("Start Measurement")
            self.export_csv(auto=True)  # Auto export when stopping

    def read_sensor(self):
        """
        Runs in a separate thread to continuously read data from the sensor.
        Emits batches of IR data to update the plot.
        Calculates BPM and status only if finger is detected on sensor.
        """
        batch_size = 10  # Number of samples per batch update
        batch = []

        while True:
            try:
                red, ir = self.sensor.read_fifo()
                batch.append(ir)

                # Emit batch for GUI plotting
                if len(batch) >= batch_size:
                    self.data_emitter.new_data.emit(batch)
                    batch.clear()

                if self.measuring:
                    timestamp = time.time()
                    readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

                    # Finger detection: only proceed if IR is above threshold
                    if ir > IR_THRESHOLD:
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
                        # Finger not detected, clear data and update labels
                        self.ir_data.clear()
                        bpm = 0
                        status = "Waiting for finger..."
                        self.label_bpm.setText("BPM: --")
                        self.label_status.setText(f"Status: {status}")

                    self.csv_data.append((timestamp, readable_time, ir, f"{bpm:.1f}", status))

                else:
                    # Finger off: do not update BPM, show message
                    self.label_bpm.setText("BPM: --")
                    self.label_status.setText("Status: Finger off")
                    # Do NOT append data or calculate BPM when finger is off

                time.sleep(0.01)
            except Exception as e:
                print(f"Error reading sensor: {e}")
                time.sleep(1)

    def update_plot(self, new_vals):
        """
        Slot to receive new IR data batch and update the plot.
        Called via Qt signal emitted from sensor thread.
        """
        self.plot.update_plot(new_vals)

    def butter_lowpass_filter(self, data, cutoff=2.5, fs=100, order=2):
        """
        Apply a Butterworth low-pass filter to the IR data.
        Args:
            data (list or np.array): input IR data samples
            cutoff (float): cutoff frequency in Hz
            fs (float): sampling frequency in Hz
            order (int): filter order
        Returns:
            np.array: filtered data
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def calc_bpm(self, ir_data, fs=100):
        """
        Calculate beats per minute (BPM) from IR data.
        Uses peak detection on filtered signal.
        Args:
            ir_data (list or np.array): IR sensor data
            fs (int): sampling frequency in Hz
        Returns:
            float: estimated BPM (0 if insufficient peaks)
        """
        filtered = self.butter_lowpass_filter(ir_data, cutoff=2.5, fs=fs, order=2)
        peaks, _ = find_peaks(filtered, distance=fs * 0.6)  # at least 0.6s between beats
        if len(peaks) < 2:
            return 0
        rr_intervals = np.diff(peaks) / fs
        bpm = 60 / np.mean(rr_intervals)
        return bpm

    def classify_bpm(self, bpm):
        """
        Classify heart rate status based on BPM value.
        Args:
            bpm (float): beats per minute
        Returns:
            str: classification string ("Bradycardia", "Tachycardia", or "Normal")
        """
        if bpm < 60:
            return "Bradycardia"
        elif bpm > 100:
            return "Tachycardia"
        else:
            return "Normal"

    def export_csv(self, auto=False):
        """
        Export recorded data to a CSV file.
        Args:
            auto (bool): if True, export automatically without user dialog
        """
        if not self.csv_data:
            return

        if auto:
            # Automatically save in user's home folder under max30102_exports/
            folder = os.path.expanduser('~/max30102_exports')
            os.makedirs(folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = os.path.join(folder, f"max30102_data_{timestamp}.csv")
        else:
            # Ask user for file location
            file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
            if not file_name:
                return

        # Write data to CSV file
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Readable Time', 'IR_Value', 'BPM', 'Status'])
            writer.writerows(self.csv_data)

        print("Data exported to:", file_name)

    def closeEvent(self, event):
        """
        Handle window close event.
        Shutdown sensor cleanly.
        """
        self.sensor.shutdown()
        event.accept()


def main():
    """
    Application entry point.
    Creates the QApplication and shows the main window.
    """
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
