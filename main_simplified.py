import sys # Provides access to system-specific parameters and functions
import csv # Used for reading and writing CSV (Comma Separated Values) files
import time # Provides time-related functions
import numpy as np # Fundamental package for numerical computation, especially with arrays
from PyQt5.QtWidgets import ( # Importing necessary widgets from PyQt5 for GUI creation
    QApplication, QMainWindow, QLabel, QVBoxLayout,
    QWidget, QPushButton, QHBoxLayout, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject # Core functionalities for Qt, signals/slots, and base object
import pyqtgraph as pg # Library for plotting data efficiently
from scipy.signal import find_peaks # Function from SciPy to find peaks in a signal
import threading # Module for multi-threading, allowing concurrent execution
from datetime import datetime # Module to work with dates and times
import os # Module to interact with the operating system, like managing file paths
from sensor.max30102 import MAX30102 # Custom module for interacting with the MAX30102 sensor


class DataEmitter(QObject):
    """
    QObject subclass used to emit batches of IR data via a PyQt signal.
    This allows the sensor reading thread to safely communicate data to the GUI thread.
    This is crucial because GUI updates must happen on the main thread.
    """
    new_data = pyqtSignal(list)  # Signal to send a batch (list) of IR data points.
                                 # When this signal is emitted, any connected slot will receive the list.


class PulsePlot(pg.PlotWidget):
    """
    Custom plot widget for displaying real-time IR sensor data.
    Uses a circular buffer to hold a fixed amount of data points, ensuring efficient
    real-time plotting without storing excessive historical data.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setBackground('w')  # Set plot background to white for better visibility.
        self.setTitle("MAX30102 Pulse Monitor", color='k', size='16pt') # Set plot title.
        self.setLabel('left', "IR Value", **{'color': 'black', 'font-size': '12pt'}) # Label for the Y-axis.
        self.setLabel('bottom', "Sample", **{'color': 'black', 'font-size': '12pt'}) # Label for the X-axis.
        self.showGrid(x=True, y=True) # Display a grid on the plot for easier data interpretation.

        self.curve = self.plot(pen=pg.mkPen(color='r', width=2))  # Create a plot curve with a red line (width 2).
        self.buffer_size = 100  # Define the number of samples to keep in the circular buffer for plotting.
        self.data = np.zeros(self.buffer_size, dtype=int)  # Initialize a NumPy array as the circular buffer for IR values.
        self.ptr = 0  # Pointer to the current position in the circular buffer, indicating where the next data point goes.

    def update_plot(self, new_vals):
        """
        Updates the plot with a new batch of IR data.
        Maintains a circular buffer of the latest samples to show a continuous waveform.
        """
        n = len(new_vals) # Get the number of new data points in the batch.
        if n > self.buffer_size:
            # If the new data batch is larger than the buffer, take only the most recent part.
            new_vals = new_vals[-self.buffer_size:]
            n = self.buffer_size

        # Logic for the circular buffer:
        # If new data fits without wrapping around to the start of the buffer:
        if self.ptr + n < self.buffer_size:
            self.data[self.ptr:self.ptr + n] = new_vals # Insert new values directly.
        else:
            # If new data needs to wrap around:
            # Fill from the current pointer to the end of the buffer.
            till_end = self.buffer_size - self.ptr
            self.data[self.ptr:] = new_vals[:till_end]
            # Fill the remaining part from the beginning of the buffer.
            self.data[:n - till_end] = new_vals[till_end:]
        self.ptr = (self.ptr + n) % self.buffer_size # Update the pointer for the next insertion.

        # Rearrange data for continuous plot display:
        # If the pointer is at the beginning, use data as is.
        if self.ptr == 0:
            display_data = self.data
        else:
            # Otherwise, concatenate the part after the pointer with the part before it
            # to show a seamless waveform as if the buffer were linearly filled.
            display_data = np.concatenate((self.data[self.ptr:], self.data[:self.ptr]))

        self.curve.setData(display_data) # Update the plot's curve with the rearranged data.


class MainApp(QMainWindow):
    """
    Main application window containing the pulse plot, labels, and buttons.
    Manages sensor reading in a separate thread and BPM calculation to keep the GUI responsive.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAX30102 Pulse Monitor") # Set window title.
        self.resize(700, 500) # Set initial window size.

        self.plot = PulsePlot(self) # Create an instance of our custom plot widget.

        # Label to display the calculated Beats Per Minute (BPM).
        self.label_bpm = QLabel("BPM: 0", alignment=Qt.AlignCenter)
        self.label_bpm.setStyleSheet("font-size: 24pt; font-weight: bold; color: #333;")

        # Label to display the heart rate status (e.g., "No Data", "No Finger", "Normal").
        self.label_status = QLabel("Status: No Data", alignment=Qt.AlignCenter)
        self.label_status.setStyleSheet("font-size: 14pt; color: #666;")

        # Button to start or stop the measurement process.
        self.btn_toggle = QPushButton("Start Measurement")
        self.btn_toggle.setStyleSheet("padding: 10px; font-size: 12pt;")
        self.btn_toggle.clicked.connect(self.toggle_measurement) # Connect button click to a method.

        # Button to export collected data to a CSV file.
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setStyleSheet("padding: 10px; font-size: 12pt;")
        self.btn_export.clicked.connect(self.export_csv) # Connect button click to a method.

        # Layout for the buttons, placing them side-by-side.
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_toggle)
        button_layout.addWidget(self.btn_export)

        # Main vertical layout for the entire window content.
        layout = QVBoxLayout()
        layout.addWidget(self.plot) # Add the plot to the layout.
        layout.addWidget(self.label_bpm) # Add the BPM label.
        layout.addWidget(self.label_status) # Add the status label.
        layout.addLayout(button_layout) # Add the button layout.

        # Create a central widget and apply the main layout to it.
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container) # Set the central widget of the QMainWindow.

        self.sensor = MAX30102() # Initialize the MAX30102 sensor instance (assuming `sensor` module is correctly set up).

        self.ir_data = []  # List to store raw IR data for BPM calculation.
        self.csv_data = []  # List to store data rows for CSV export (timestamp, IR, BPM, status).

        self.measuring = False # Flag to indicate whether measurement is currently active.
        self.data_emitter = DataEmitter() # Create an instance of DataEmitter for cross-thread communication.
        self.data_emitter.new_data.connect(self.update_plot) # Connect the emitter's signal to the plot update slot.
        self.running = True # Flag to control the sensor reading thread's loop.

        # Create and start a separate thread for reading sensor data.
        # This prevents the GUI from freezing while reading data.
        self.thread = threading.Thread(target=self.read_sensor)
        self.thread.daemon = True # Set as a daemon thread, so it terminates when the main program exits.
        self.thread.start() # Start the sensor reading thread.

        self.SAMPLING_RATE = 100  # Define the sampling frequency of the sensor in Hz (samples per second).
        self.IR_THRESHOLD = 50000  # A threshold value for IR data to detect if a finger is placed on the sensor.
                                   # Values below this might indicate no finger or poor contact.


    def toggle_measurement(self):
        """
        Starts or stops pulse measurement.
        Clears previous data when starting a new measurement.
        Exports data to CSV automatically when stopping.
        """
        self.measuring = not self.measuring # Toggle the measurement flag.
        if self.measuring:
            self.ir_data.clear() # Clear accumulated IR data.
            self.csv_data.clear() # Clear data prepared for CSV export.
            self.label_status.setText("Status: Measuring...") # Update status label.
            self.btn_toggle.setText("Stop Measurement") # Change button text.
        else:
            self.btn_toggle.setText("Start Measurement") # Change button text.
            self.export_csv(auto=True) # Automatically export data when measurement stops.

    def read_sensor(self):
        """
        Runs in a separate thread to continuously read data from the MAX30102 sensor.
        Processes batches of data for plotting and calculates BPM if measurement is active.
        """
        batch_size = 10 # Number of samples to collect before emitting a batch to the GUI.
        batch = [] # Temporary list to hold current batch of IR data.

        while self.running: # Loop as long as the application is running.
            try:
                red, ir = self.sensor.read_fifo() # Read Red and IR values from the sensor's FIFO buffer.
                if ir is not None: # Ensure a valid IR reading was obtained.
                    batch.append(ir) # Add the IR value to the current batch.

                    if len(batch) >= batch_size:
                        self.data_emitter.new_data.emit(batch) # Emit the batch to update the plot.
                        batch.clear() # Clear the batch for the next set of readings.

                    if self.measuring: # Only process for BPM calculation if measurement is active.
                        timestamp = time.time() # Get current Unix timestamp.
                        readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') # Format timestamp for readability.

                        self.ir_data.append(ir) # Add the current IR sample to the main data list.
                        # Keep only the last 5 seconds of data for BPM calculation to avoid accumulating too much data.
                        # (5 seconds * 100 samples/second = 500 samples)
                        if len(self.ir_data) > self.SAMPLING_RATE * 5:
                            self.ir_data = self.ir_data[-self.SAMPLING_RATE * 5:]

                        bpm = 0
                        status = "No Finger" # Default status if no valid signal.

                        # Check for finger presence based on the IR_THRESHOLD.
                        # If the max IR value in the last batch is below the threshold, assume no finger.
                        if max(self.ir_data[-min(batch_size, len(self.ir_data)):]) < self.IR_THRESHOLD:
                            status = "No Finger"
                            bpm = 0
                        # If enough data is collected (at least 3 seconds) and a finger is likely present:
                        elif len(self.ir_data) >= self.SAMPLING_RATE * 3:
                            bpm = self.calc_bpm(self.ir_data) # Calculate BPM.
                            # Basic classification of heart rate status.
                            status = "Normal" if 40 < bpm < 100 else ("Bradycardia" if bpm <= 40 else "Tachycardia")

                        self.label_bpm.setText(f"BPM: {bpm:.1f}") # Update BPM display.
                        self.label_status.setText(f"Status: {status}") # Update status display.

                        # Append current data point to the CSV export list.
                        self.csv_data.append((timestamp, readable_time, ir, f"{bpm:.1f}", status))

                time.sleep(0.01) # Small delay to yield control and prevent the thread from consuming 100% CPU.

            except Exception as e:
                # Print any errors encountered during sensor reading.
                print(f"Error reading sensor: {e}")
                time.sleep(1) # Wait longer on error to prevent rapid error logging and resource hogging.

    def update_plot(self, new_vals):
        """
        Slot to update the plot widget with new IR data.
        This method is connected to the `new_data` signal of `DataEmitter`.
        """
        self.plot.update_plot(new_vals) # Call the plot's own update method.

    def calc_bpm(self, ir_data, fs=100):
        """
        Calculates BPM from IR data using basic peak detection.
        This simplified version relies less on advanced signal processing (like complex filters).
        """
        data = np.array(ir_data) # Convert the input list to a NumPy array for efficient numerical operations.

        # In this simplified version, we don't apply complex band-pass filters.
        # We only perform basic detrending to remove the baseline drift from the signal.
        # Detrending makes the signal oscillate around zero, which helps `find_peaks`.
        detrended_data = data - np.mean(data)

        # Find peaks in the detrended data.
        # `distance`: Minimum number of samples between consecutive peaks.
        # A value of `fs * 0.6` (0.6 seconds at 100Hz = 60 samples) is chosen as a reasonable
        # minimum time between two human heartbeats, preventing detection of spurious peaks within one beat.
        peaks, _ = find_peaks(detrended_data, distance=int(fs * 0.6))

        if len(peaks) < 2:
            return 0  # Return 0 if fewer than 2 peaks are found, as BPM calculation requires at least two peaks.

        # Calculate the time intervals between consecutive peaks (RR intervals) in seconds.
        rr_intervals = np.diff(peaks) / fs
        # Calculate BPM by converting the average RR interval to beats per minute (60 seconds / average interval).
        bpm = 60 / np.mean(rr_intervals)
        return bpm

    def export_csv(self, auto=False):
        """
        Exports collected measurement data to a CSV file.
        Can either save automatically to a default timestamped file or prompt the user for a filename.
        """
        if not self.csv_data: # Check if there is any data to export.
            print("No data to export.")
            return

        if auto: # If auto-export is requested (e.g., when measurement stops).
            folder = os.path.expanduser('~/max30102_exports') # Define a default export folder in the user's home directory.
            os.makedirs(folder, exist_ok=True) # Create the folder if it doesn't exist.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Generate a timestamp for the filename.
            file_name = os.path.join(folder, f"max30102_data_{timestamp}.csv") # Construct the full file path.
        else: # If manual export is requested (user clicks "Export CSV").
            file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)") # Open a file save dialog.
            if not file_name: # If the user cancels the dialog.
                return

        try:
            with open(file_name, 'w', newline='') as file: # Open the CSV file in write mode.
                writer = csv.writer(file) # Create a CSV writer object.
                writer.writerow(['Timestamp', 'Readable Time', 'IR_Value', 'BPM', 'Status']) # Write the header row.
                writer.writerows(self.csv_data) # Write all collected data rows.
            print("Data exported to:", file_name)
        except IOError as e: # Catch any errors during file writing.
            print(f"Could not save file: {e}")

    def closeEvent(self, event):
        """
        Handles the application close event.
        Ensures the sensor reading thread is properly shut down before the application exits.
        """
        self.running = False # Set the flag to stop the sensor reading thread's loop.
        if self.thread.is_alive(): # Check if the thread is still running.
            self.thread.join(timeout=1) # Wait for the thread to finish, with a 1-second timeout.
                                       # This ensures a clean exit if the thread is busy.
        self.sensor.shutdown() # Call the sensor's shutdown method (specific to your MAX30102 class).
        event.accept() # Accept the close event, allowing the application to terminate.


def main():
    """
    Main function to create and run the application.
    """
    app = QApplication(sys.argv) # Create a QApplication instance, required for any PyQt5 application.
    window = MainApp() # Create an instance of our main application window.
    window.show() # Display the main window.
    sys.exit(app.exec_()) # Start the PyQt event loop and exit the application when it finishes.


if __name__ == "__main__":
    main() # Run the main function when the script is executed.