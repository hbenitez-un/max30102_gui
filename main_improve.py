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


class DataEmitter(QObject):
    """
    QObject subclass used to emit new IR data batches via a PyQt signal.
    This allows the sensor reading thread to communicate data to the GUI thread safely.
    """
    new_data = pyqtSignal(list)  # Signal to send a batch (list) of IR data points


class PulsePlot(pg.PlotWidget):
    """
    Custom plot widget for displaying IR sensor data in real-time.
    Uses a circular buffer to hold a fixed amount of data points.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setBackground('w')  # White background
        self.setTitle("MAX30102 Pulse Monitor", color='k', size='16pt')
        self.setLabel('left', "IR Value", **{'color': 'black', 'font-size': '12pt'})
        self.setLabel('bottom', "Sample", **{'color': 'black', 'font-size': '12pt'})
        self.showGrid(x=True, y=True)

        self.curve = self.plot(pen=pg.mkPen(color='r', width=2))  # Red line for data
        self.buffer_size = 100  # Number of samples to keep in buffer
        self.data = np.zeros(self.buffer_size, dtype=int)  # Circular buffer for IR values
        self.ptr = 0  # Current index in the buffer

    def update_plot(self, new_vals):
        """
        Update the plot with a new batch of IR data.
        Maintains a circular buffer of the latest samples.
        """
        n = len(new_vals)
        if n > self.buffer_size:
            # If new data is bigger than buffer, keep only the most recent part
            new_vals = new_vals[-self.buffer_size:]
            n = self.buffer_size

        if self.ptr + n < self.buffer_size:
            # If new data fits without wrapping, insert directly
            self.data[self.ptr:self.ptr + n] = new_vals
        else:
            # Otherwise, split data to wrap around the buffer end
            till_end = self.buffer_size - self.ptr
            self.data[self.ptr:] = new_vals[:till_end]
            self.data[:n - till_end] = new_vals[till_end:]
        self.ptr = (self.ptr + n) % self.buffer_size

        # Rearrange data so plot shows continuous waveform
        if self.ptr == 0:
            display_data = self.data
        else:
            display_data = np.concatenate((self.data[self.ptr:], self.data[:self.ptr]))

        self.curve.setData(display_data)


class MainApp(QMainWindow):
    """
    Main application window that contains the pulse plot, labels, and buttons.
    Handles sensor reading in a separate thread and BPM calculation.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAX30102 Pulse Monitor")
        self.resize(700, 500)

        # Create and add the pulse plot widget
        self.plot = PulsePlot(self)

        # Label to display BPM (beats per minute)
        self.label_bpm = QLabel("BPM: 0", alignment=Qt.AlignCenter)
        self.label_bpm.setStyleSheet("font-size: 24pt; font-weight: bold; color: #333;")

        # Label to display heart rate status (normal, bradycardia, tachycardia)
        self.label_status = QLabel("Status: Normal", alignment=Qt.AlignCenter)
        self.label_status.setStyleSheet("font-size: 14pt; color: #666;")

        # Button to start/stop the measurement
        self.btn_toggle = QPushButton("Start Measurement")
        self.btn_toggle.setStyleSheet("padding: 10px; font-size: 12pt;")
        self.btn_toggle.clicked.connect(self.toggle_measurement)

        # Button to export collected data to CSV
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setStyleSheet("padding: 10px; font-size: 12pt;")
        self.btn_export.clicked.connect(self.export_csv)

        # Layout for the buttons side-by-side
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_toggle)
        button_layout.addWidget(self.btn_export)

        # Main vertical layout
        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        layout.addWidget(self.label_bpm)
        layout.addWidget(self.label_status)
        layout.addLayout(button_layout)

        # Set central widget with the layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initialize MAX30102 sensor instance
        self.sensor = MAX30102()

        # List to store IR data for BPM calculation
        self.ir_data = []

        # List to store data rows for CSV export
        self.csv_data = []

        # Flag indicating whether measurement is active
        self.measuring = False

        # Create data emitter for thread-safe signal emission
        self.data_emitter = DataEmitter()
        self.data_emitter.new_data.connect(self.update_plot)
        self.running = True  # <-- flag to control the sensor thread

        # Start the sensor reading thread
        self.thread = threading.Thread(target=self.read_sensor)
        self.thread.daemon = True
        self.thread.start()

        self.bpm_history = []  # For BPM smoothing
        self.IR_THRESHOLD = 50000  # Adjust if needed for finger detection
        self.SAMPLING_RATE = 100 # Hz. As per MAX30102 typical configuration


    def toggle_measurement(self):
        """
        Start or stop measuring pulse.
        Clears data when starting; exports CSV when stopping.
        """
        self.measuring = not self.measuring
        if self.measuring:
            self.ir_data.clear()
            self.csv_data.clear()
            self.bpm_history.clear() # Clear BPM history when starting a new measurement - MEJORA
            self.btn_toggle.setText("Stop Measurement")
        else:
            self.btn_toggle.setText("Start Measurement")
            self.export_csv(auto=True)

    def read_sensor(self):
        batch_size = 10
        batch = []

        while self.running:
            try:
                red, ir = self.sensor.read_fifo() # Red is for SPO2
                if ir is not None: # Ensure a valid IR reading - MEJORA
                    batch.append(ir)

                    if len(batch) >= batch_size:
                        self.data_emitter.new_data.emit(batch)
                        batch.clear()

                    if self.measuring:
                        timestamp = time.time()
                        readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

                        # Add new IR sample
                        self.ir_data.append(ir)
                        # Keep a window of data, e.g., 10 seconds worth for BPM calculation - MEJORA
                        # At 100Hz, 10 seconds = 1000 samples
                        if len(self.ir_data) > self.SAMPLING_RATE * 10:
                            self.ir_data = self.ir_data[-self.SAMPLING_RATE * 10:]

                        bpm = 0
                        status = "No data"

                        # Check for valid IR signal (finger detection)
                        if max(self.ir_data[-min(batch_size, len(self.ir_data)):]) < self.IR_THRESHOLD:
                            status = "No finger detected"
                            bpm = 0
                            self.bpm_history.clear() # Clear BPM history if finger is removed - MEJORA
                        elif len(self.ir_data) >= self.SAMPLING_RATE * 4: # At least 4 seconds of data for more stable BPM
                            bpm = self.calc_bpm(self.ir_data)
                            status = self.classify_bpm(bpm)

                            # Smooth BPM using moving average of last 10 values for more stability - MEJORA
                            if bpm > 0: # Only add valid BPMs to history
                                self.bpm_history.append(bpm)
                                if len(self.bpm_history) > 10: # Increased window for smoothing
                                    self.bpm_history.pop(0)
                                bpm = sum(self.bpm_history) / len(self.bpm_history)
                            else:
                                self.bpm_history.clear() # Clear if current BPM is 0 - MEJORA

                        self.label_bpm.setText(f"BPM: {bpm:.1f}")
                        self.label_status.setText(f"Status: {status}")

                        self.csv_data.append((timestamp, readable_time, ir, f"{bpm:.1f}", status))

                time.sleep(0.01) # Small delay to avoid busy-waiting and allow other threads to run - MEJORA

            except Exception as e:
                print(f"Error reading sensor: {e}")
                time.sleep(1) # Wait longer on error to prevent rapid error logging - MEJORA


    def update_plot(self, new_vals):
        """
        Slot to update the plot widget with new IR data.
        """
        self.plot.update_plot(new_vals)

    def butter_bandpass_filter(self, data, lowcut=0.7, highcut=3.5, fs=100, order=4):
        """
        Applies a Butterworth band-pass filter to the IR data.
        Adjusted highcut and order for better pulse signal isolation. - MEJORA
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def calc_bpm(self, ir_data, fs=100):
        """
        Calculate beats per minute (BPM) from IR data using peak detection.
        Includes signal quality checks and refined peak detection parameters. - MEJORA
        :param ir_data: list or numpy array of IR data samples
        :param fs: sampling frequency in Hz (default 100 Hz)
        :return: calculated BPM as float
        """
        data = np.array(ir_data)

        # Detrend the signal to remove baseline drift - MEJORA
        detrended_data = data - np.mean(data)

        # Apply band-pass filter
        filtered_data = self.butter_bandpass_filter(detrended_data, lowcut=0.7, highcut=3.5, fs=fs, order=4)

        # Normalize the filtered data to help with peak detection consistency - MEJORA
        # Avoid division by zero if all values are the same
        if np.max(filtered_data) - np.min(filtered_data) > 0:
            normalized_data = (filtered_data - np.min(filtered_data)) / (np.max(filtered_data) - np.min(filtered_data))
        else:
            return 0 # No variance in signal, likely flatline

        # Detect peaks.
        # `distance`: Minimum number of samples between peaks. - MEJORA
        # `prominence`: Required prominence of peaks. - MEJORA
        peaks, properties = find_peaks(normalized_data, distance=int(fs * 0.35), prominence=0.2)

        if len(peaks) < 2:
            return 0  # Not enough peaks to calculate BPM

        # Calculate RR intervals in seconds between detected peaks
        rr_intervals = np.diff(peaks) / fs

        # Optional: Filter out outliers in RR intervals (e.g., intervals too short or too long) - MEJORA
        # This helps improve robustness against spurious peak detections
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        # Keep intervals within 2 standard deviations of the mean
        valid_rr_intervals = rr_intervals[(rr_intervals > (mean_rr - 2 * std_rr)) & (rr_intervals < (mean_rr + 2 * std_rr))]

        if len(valid_rr_intervals) < 1:
            return 0 # No valid intervals after filtering

        bpm = 60 / np.mean(valid_rr_intervals)  # Convert average interval to BPM
        return bpm


    def classify_bpm(self, bpm):
        """
        Classify the heart rate status based on BPM value.
        :param bpm: beats per minute
        :return: string status ("Bradycardia", "Tachycardia", or "Normal")
        """
        # Consider a wider 'normal' range for general applications - MEJORA
        if bpm < 50:
            return "Bradycardia"
        elif bpm > 100:
            return "Tachycardia"
        elif bpm == 0: # Handle 0 BPM explicitly - MEJORA
            return "Invalid"
        else:
            return "Normal"

    def export_csv(self, auto=False):
        """
        Export the collected measurement data to a CSV file.
        If auto=True, saves to a default folder with timestamped filename.
        Otherwise, opens a save dialog.
        """
        if not self.csv_data:
            print("No data to export.") # Inform user if no data - MEJORA
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

        try: # Added try-except for file operations - MEJORA
            with open(file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Readable Time', 'IR_Value', 'BPM', 'Status'])
                writer.writerows(self.csv_data)
            print("Data exported to:", file_name)
        except IOError as e:
            print(f"Could not save file: {e}")


    def closeEvent(self, event):
        """
        Handles application close event.
        Properly shuts down the sensor.
        """
        self.running = False  # <-- stop thread loop
        if self.thread.is_alive():
            self.thread.join(timeout=1) # Wait for thread to finish, with a timeout - MEJORA
        self.sensor.shutdown()
        event.accept()


def main():
    """
    Main function to create and run the application.
    """
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()