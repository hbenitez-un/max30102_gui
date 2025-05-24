import sys
import csv
import time
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout,
    QWidget, QPushButton, QHBoxLayout, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer
import pyqtgraph as pg
from scipy.signal import find_peaks, butter, filtfilt
import threading
from datetime import datetime
import os

# Ensures the correct import for the MAX30102 sensor class,
# adapting to potential variations in its packaging or file location.
# This section attempts to import the class based on common setups,
# supporting data returned as a tuple (red, IR) or a dictionary.
try:
    # Attempts import if MAX30102 class is within a 'sensor' package.
    from sensor.max30102 import MAX30102
except ImportError:
    # If the initial import fails, attempts direct import from a file
    # named 'max30102.py' or from an installed library like 'max30102_sensor'.
    try:
        from max30102 import MAX30102
    except ImportError:
        try:
            from max30102_sensor import MAX30102
        except ImportError:
            # If all import attempts fail, an error message is printed,
            # and the application exits.
            print("Error: The MAX30102 class could not be imported.")
            print("Verify that the sensor library is installed or 'max30102.py' is accessible.")
            sys.exit(1)


# --- Configuration Constants ---
# These constants are configured to match the sensor's specifications
# and the application's operational requirements.
# SAMPLE_FREQ: Represents the actual sampling frequency of the MAX30102 sensor.
# This value is critical for accurate BPM calculations and should align with
# the sensor's documentation (common values include 25, 50, 100 Hz).
SAMPLE_FREQ = 100 # Hz (e.g., 100 Hz is a common setting for the MAX30102)

# BUFFER_SIZE_PLOT: Defines the number of samples displayed in the real-time graph.
# This value determines the visual time window of the displayed data.
BUFFER_SIZE_PLOT = SAMPLE_FREQ * 5 # Configured to display 5 seconds of data

# BUFFER_SIZE_BPM: Specifies the number of samples utilized for BPM calculation.
# A duration of approximately 4-6 seconds of data is typically effective
# for ensuring a stable heart rate measurement.
BUFFER_SIZE_BPM = SAMPLE_FREQ * 4 # Uses 4 seconds of data for BPM calculation

# IR_THRESHOLD: Sets the minimum infrared (IR) value required to detect
# the presence of a finger on the sensor. This threshold is typically
# calibrated by observing IR values with and without a finger placed on the sensor.
IR_THRESHOLD = 15000 # Minimum IR value to register a finger presence


class DataEmitter(QObject):
    """
    This QObject subclass is designed to facilitate the safe emission
    of new batches of IR data via PyQt signals. This mechanism ensures
    that data can be communicated securely from the sensor reading thread
    to the GUI thread, maintaining thread safety.
    """
    new_data = pyqtSignal(list)  # Signal to transmit a list of IR data points for plotting.
    new_bpm_status = pyqtSignal(float, str) # Signal to update the displayed BPM and status string.

class PulsePlot(pg.PlotWidget):
    """
    A custom plot widget implemented using pyqtgraph, specifically for
    displaying real-time infrared (IR) sensor data. This widget utilizes
    a circular buffer to efficiently manage and retain a fixed quantity
    of incoming data points for continuous visualization.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setBackground('w')  # Sets the plot background to white.
        self.setTitle("MAX30102 Pulse Monitor", color='k', size='16pt') # Configures plot title and style.
        self.setLabel('left', "IR Value", **{'color': 'black', 'font-size': '12pt'}) # Sets the left axis label.
        self.setLabel('bottom', "Sample", **{'color': 'black', 'font-size': '12pt'}) # Sets the bottom axis label.
        self.showGrid(x=True, y=True) # Displays a grid on both axes.

        # Establishes the primary curve for displaying raw IR data.
        self.curve = self.plot(pen=pg.mkPen(color='r', width=2), name="Raw IR") 
        # An optional curve for visualizing filtered IR data, useful for debugging.
        # self.curve_filtered = self.plot(pen=pg.mkPen(color='b', width=1), name="Filtered IR") 

        self.buffer_size = BUFFER_SIZE_PLOT  # The number of samples maintained in the buffer for plotting.
        self.data = np.zeros(self.buffer_size, dtype=int)  # Initializes a circular buffer for IR values.
        self.ptr = 0  # The current index within the circular buffer.

    def update_plot(self, new_vals):
        """
        Updates the plot with a new batch of IR data. This method efficiently
        manages a circular buffer, ensuring that only the most recent samples
        are retained and displayed, providing a continuous waveform visualization.
        """
        n = len(new_vals)
        if n == 0:
            return

        # If the incoming data batch exceeds the buffer size, only the most
        # recent portion of the data is retained for display.
        if n > self.buffer_size:
            new_vals = new_vals[-self.buffer_size:]
            n = self.buffer_size

        if self.ptr + n <= self.buffer_size:
            # If the new data fits within the current buffer segment without
            # wrapping, it is inserted directly.
            self.data[self.ptr:self.ptr + n] = new_vals
        else:
            # If the new data extends beyond the end of the buffer, it is
            # split and wrapped around, filling the remainder of the buffer
            # and then starting from the beginning.
            till_end = self.buffer_size - self.ptr
            self.data[self.ptr:] = new_vals[:till_end]
            self.data[:n - till_end] = new_vals[till_end:]
        self.ptr = (self.ptr + n) % self.buffer_size # Updates the buffer pointer.

        # Rearranges the data within the buffer to present a continuous waveform
        # on the plot, regardless of the circular buffer's internal state.
        if self.ptr == 0:
            display_data = self.data
        else:
            display_data = np.concatenate((self.data[self.ptr:], self.data[:self.ptr]))

        self.curve.setData(display_data) # Updates the plot with the prepared data.

        # (Optional) If `self.curve_filtered` is uncommented, this section
        # would update it, requiring filtered data to be passed to the plot.
        # self.curve_filtered.setData(display_filtered_data)


class MainApp(QMainWindow):
    """
    The MainApp class represents the principal application window,
    integrating the pulse graph, status labels, and control buttons.
    It manages the reading of sensor data in a dedicated separate thread
    and performs heart rate (BPM) calculations.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAX30102 Pulse Monitor") # Sets the window title.
        self.resize(700, 500) # Sets the initial window size.

        # Initializes and incorporates the custom pulse plot widget.
        self.plot = PulsePlot(self)

        # Configures the QLabel for displaying the calculated BPM.
        self.label_bpm = QLabel("BPM: 0", alignment=Qt.AlignCenter)
        self.label_bpm.setStyleSheet("font-size: 24pt; font-weight: bold; color: #333;")

        # Configures the QLabel for displaying the heart rate status.
        self.label_status = QLabel("Status: Initializing sensor...", alignment=Qt.AlignCenter)
        self.label_status.setStyleSheet("font-size: 14pt; color: #666;")

        # Initializes the QPushButton for toggling measurement.
        self.btn_toggle = QPushButton("Start Measurement")
        self.btn_toggle.setStyleSheet("padding: 10px; font-size: 12pt;")
        self.btn_toggle.clicked.connect(self.toggle_measurement) # Connects to the toggle method.

        # Initializes the QPushButton for exporting data to CSV.
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setStyleSheet("padding: 10px; font-size: 12pt;")
        self.btn_export.clicked.connect(self.export_csv) # Connects to the export method.

        # Arranges the control buttons horizontally using QHBoxLayout.
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_toggle)
        button_layout.addWidget(self.btn_export)

        # Establishes the main vertical layout for the application's widgets.
        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        layout.addWidget(self.label_bpm)
        layout.addWidget(self.label_status)
        layout.addLayout(button_layout)

        # Sets the central widget of the QMainWindow to the container
        # holding the arranged layout.
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initializes the MAX30102 sensor instance. Error handling is included
        # to manage potential initialization failures.
        self.sensor = None # Sensor instance is initialized to None.
        try:
            self.sensor = MAX30102()
            print("MAX30102 sensor successfully initialized.")
            self.label_status.setText("Status: Waiting for finger...")
        except Exception as e:
            print(f"Error encountered during MAX30102 sensor initialization: {e}")
            self.label_status.setText("Error: Sensor not initialized. Verify connections and libraries.")
            self.btn_toggle.setEnabled(False) # Disables the toggle button if sensor initialization fails.

        # Initializes lists for storing IR data for BPM calculation and CSV export.
        self.ir_data = [] # Stores IR data for BPM calculation.
        self.csv_data = [] # Stores data rows for CSV export.
        self.measuring = False # Flag indicating the active state of measurement.

        # Creates a DataEmitter instance to facilitate thread-safe signal emission.
        self.data_emitter = DataEmitter()
        # Connects the `new_data` signal to the plot's update function.
        self.data_emitter.new_data.connect(self.plot.update_plot)
        # Connects the `new_bpm_status` signal to the label update function.
        self.data_emitter.new_bpm_status.connect(self.update_bpm_status_labels)

        self.running = True  # Flag to control the sensor thread's operational loop.
        
        # Initiates the sensor reading thread only if the sensor
        # was successfully initialized.
        if self.sensor:
            self.thread = threading.Thread(target=self.read_sensor)
            self.thread.daemon = True # Ensures the thread terminates with the main application.
            self.thread.start()

        self.bpm_history = []  # Stores historical BPM values for smoothing.
        
        # Initializes a QTimer to periodically update the BPM status and GUI.
        # This ensures that GUI updates are executed safely within the main thread.
        self.bpm_timer = QTimer(self)
        self.bpm_timer.setInterval(500) # Configures the timer to update every 500 ms.
        self.bpm_timer.timeout.connect(self.calculate_and_update_bpm) # Connects to the calculation method.
        self.bpm_timer.start() # Starts the timer.


    def toggle_measurement(self):
        """
        Toggles the pulse measurement state between active and inactive.
        Upon initiation, it clears previous data; upon cessation, it triggers
        an automatic CSV export.
        """
        if not self.sensor: # Prevents measurement initiation if the sensor is not initialized.
            return

        self.measuring = not self.measuring # Inverts the measurement state.
        if self.measuring:
            self.ir_data.clear() # Clears accumulated IR data.
            self.csv_data.clear() # Clears data prepared for CSV export.
            self.bpm_history.clear() # Clears BPM history for fresh smoothing.
            self.label_bpm.setText("BPM: 0") # Resets BPM display.
            self.label_status.setText("Status: Measuring...") # Updates status to "Measuring".
            self.btn_toggle.setText("Stop Measurement") # Changes button text to "Stop Measurement".
        else:
            self.btn_toggle.setText("Start Measurement") # Changes button text back to "Start Measurement".
            self.export_csv(auto=True) # Automatically exports data to CSV upon stopping.
            self.label_status.setText("Status: Measurement stopped") # Updates status to "Measurement stopped".
            self.label_bpm.setText("BPM: 0") # Resets BPM display.


    def read_sensor(self):
        """
        Executes in a separate thread, continuously reading data from the MAX30102 sensor.
        It emits batches of data for real-time plotting and accumulates data
        for subsequent BPM calculations.
        """
        batch_size = 10 # Number of samples to be sent to the plot in each update cycle.
        batch = [] # Temporary list to accumulate samples before emission.
        
        # Calculates the required delay to approximate the desired sampling frequency.
        read_delay = 1.0 / SAMPLE_FREQ # E.g., 1.0 / 100 Hz results in a 0.01-second delay.

        while self.running: # Loop continues as long as the 'running' flag is True.
            try:
                # Attempts to read data from the sensor's FIFO buffer. The method
                # `self.sensor.read_fifo()` is expected to return either a tuple
                # (red_value, ir_value) or a dictionary {'red': ..., 'ir': ...}.
                fifo_data = self.sensor.read_fifo()
                
                if fifo_data is None:
                    # If no data is returned (e.g., empty FIFO or internal sensor error),
                    # the thread pauses briefly before attempting to read again.
                    time.sleep(read_delay / 2)
                    continue # Skips to the next iteration of the loop.

                # --- Adaptation for read_fifo() data format ---
                ir = 0
                if isinstance(fifo_data, tuple) and len(fifo_data) >= 2:
                    # Assumes the data format is (red_value, ir_value), extracting the IR component.
                    ir = fifo_data[1]
                elif isinstance(fifo_data, dict) and 'ir' in fifo_data:
                    # Assumes the data format is {'red': ..., 'ir': ...}, extracting the IR component.
                    ir = fifo_data['ir']
                else:
                    # Logs a warning if an unexpected data format is received and skips the current cycle.
                    print(f"Warning: read_fifo() returned an unexpected format: {type(fifo_data)} - {fifo_data}. Skipping cycle.")
                    time.sleep(read_delay / 2)
                    continue

                # Filters out extremely low IR values, which are typically indicative of noise.
                if ir < 100:
                    ir = 0 # Sets the value to 0 to ignore noise readings.

                batch.append(ir) # Adds the processed IR value to the current batch.

                if self.measuring:
                    # Accumulates IR data for BPM calculation and CSV export.
                    self.ir_data.append(ir)
                    # Maintains `ir_data` buffer at a manageable size for efficient BPM processing.
                    if len(self.ir_data) > BUFFER_SIZE_BPM + SAMPLE_FREQ * 2:
                           self.ir_data = self.ir_data[-BUFFER_SIZE_BPM - SAMPLE_FREQ * 2:]

                # Emits the collected batch of data to the plot when it reaches the specified size.
                if len(batch) >= batch_size:
                    self.data_emitter.new_data.emit(batch)
                    batch = [] # Resets the batch after emission.

                time.sleep(read_delay) # Pauses to control the sensor reading frequency.

            except Exception as e:
                # Catches any exceptions occurring during sensor reading, printing an error message.
                # The GUI status is updated to reflect a sensor failure, and the thread pauses before retrying.
                print(f"Error in sensor reading thread: {e}")
                self.label_status.setText("Error: Sensor failure. Retrying...")
                time.sleep(2)


    def update_bpm_status_labels(self, bpm, status):
        """
        This method serves as a slot for updating the BPM and status labels
        on the GUI. It is invoked via a signal from the QTimer, ensuring that
        all GUI updates occur safely within the main thread.
        """
        self.label_bpm.setText(f"BPM: {bpm:.1f}") # Updates the BPM display, formatted to one decimal place.
        self.label_status.setText(f"Status: {status}") # Updates the status message.

    def calculate_and_update_bpm(self):
        """
        Calculates the BPM and periodically updates the status labels.
        This function is invoked by a QTimer to guarantee that all GUI updates
        are executed safely on the main thread, preventing concurrency issues.
        """
        if not self.measuring:
            # If measurement is not active, BPM calculation and updates are skipped.
            self.data_emitter.new_bpm_status.emit(0.0, "Measurement stopped")
            return

        current_ir_data = np.array(self.ir_data, dtype=float) # Converts accumulated IR data to a float array for processing.

        bpm = 0.0
        status = "Collecting data..."

        # 1. Checks for finger presence by evaluating recent IR data.
        # It verifies if a sufficient amount of data has been collected and
        # if the average of recent IR readings exceeds the defined threshold.
        if len(current_ir_data) < SAMPLE_FREQ: # Requires at least 1 second of data for initial checks.
            status = "Collecting data..."
        elif np.mean(current_ir_data[-SAMPLE_FREQ:]) < IR_THRESHOLD: # Compares recent IR average to the threshold.
            status = "No finger detected"
            self.bpm_history.clear() # Clears BPM history if no finger is detected.
        else:
            # 2. Calculates BPM if sufficient data is available and a finger is detected.
            if len(current_ir_data) >= BUFFER_SIZE_BPM:
                # Processes a segment of the most recent data for BPM calculation.
                segment_to_process = current_ir_data[-BUFFER_SIZE_BPM:]
                
                bpm = self.calc_bpm(segment_to_process) # Calls the method to calculate BPM.
                
                # 3. Smooths the calculated BPM if it falls within a reasonable range (40-180 BPM)
                # and classifies the heart rate status.
                if 40 <= bpm <= 180:
                    self.bpm_history.append(bpm) # Adds the current BPM to history.
                    if len(self.bpm_history) > 10: # Keeps only the last 10 values for smoothing.
                        self.bpm_history.pop(0)
                    bpm = sum(self.bpm_history) / len(self.bpm_history) # Calculates the smoothed BPM.
                    status = self.classify_bpm(bpm) # Classifies the smoothed BPM.
                else:
                    self.bpm_history.clear() # Resets BPM history if the calculated BPM is invalid.
                    status = "Analyzing signal..." # Updates status, e.g., "BPM out of range."
                    bpm = 0.0 # Resets BPM to 0 if invalid.
            else:
                status = "Collecting data..." # Continues to collect data if buffer is insufficient.

        # Emits the signal to update the GUI labels, ensuring thread-safe operation via QTimer.
        self.data_emitter.new_bpm_status.emit(bpm, status)

        # 4. Stores data in CSV format only when measurement is active and valid data exists.
        if self.measuring and len(self.ir_data) > 0:
            timestamp = time.time() # Records the current Unix timestamp.
            readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') # Formats timestamp to readable string.
            self.csv_data.append((timestamp, readable_time, self.ir_data[-1], f"{bpm:.1f}", status)) # Appends data for CSV export.

    def butter_lowpass_filter(self, data, cutoff=2.5, fs=SAMPLE_FREQ, order=2):
        """
        Applies a Butterworth low-pass filter to the provided IR data.
        This filtering operation is beneficial for smoothing the signal
        and mitigating high-frequency noise.
        """
        nyq = 0.5 * fs  # Calculates the Nyquist frequency.
        normal_cutoff = cutoff / nyq # Normalizes the cutoff frequency.
        b, a = butter(order, normal_cutoff, btype='low', analog=False) # Designs the filter.
        y = filtfilt(b, a, data) # Applies the filter to the data.
        return y

    def butter_bandpass_filter(self, data, lowcut=0.7, highcut=3.0, fs=SAMPLE_FREQ, order=4):
        """
        Applies a Butterworth band-pass filter to the IR data.
        This is a critical step for isolating the pulse frequency,
        typically ranging from 0.7 Hz to 3.0 Hz, which corresponds to
        human heart rates.
        """
        nyq = 0.5 * fs # Calculates the Nyquist frequency.
        low = lowcut / nyq # Normalizes the lower cutoff frequency.
        high = highcut / nyq # Normalizes the higher cutoff frequency.
        b, a = butter(order, [low, high], btype='band') # Designs the band-pass filter.
        y = filtfilt(b, a, data) # Applies the filter to the data.
        return y

    
    def classify_bpm(self, bpm):
        """
        Classifies the heart rate status based on the provided BPM value.
        Categories include "No reading," "Bradycardia," "Tachycardia," and "Normal,"
        based on standard heart rate ranges.
        """
        if bpm == 0:
            return "No reading"
        elif bpm < 60:
            return "Bradycardia"
        elif bpm > 90:
            return "Tachycardia"
        else:
            return "Normal"

    def export_csv(self, auto=False):
        """
        Exports the collected measurement data to a CSV file.
        If `auto` is True, the file is saved automatically in a
        predetermined folder with a timestamped filename.
        Otherwise, a file dialog is presented to the user for selection.
        """
        if not self.csv_data:
            print("No data available for export.")
            return

        if auto:
            # Defines a default export folder within the user's home directory.
            folder = os.path.expanduser('~/max30102_exports')
            os.makedirs(folder, exist_ok=True) # Creates the directory if it does not exist.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Generates a timestamp for the filename.
            file_name = os.path.join(folder, f"max30102_data_{timestamp}.csv") # Constructs the full file path.
        else:
            # Opens a file dialog for the user to select the save location and filename.
            file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
            if not file_name:
                return # Exits if no file name is selected.

        try:
            with open(file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Unix Timestamp', 'Readable Time', 'IR Value', 'BPM', 'Status']) # Writes the CSV header.
                writer.writerows(self.csv_data) # Writes all collected data rows.
            print(f"Data successfully exported to: {file_name}")
        except Exception as e:
            print(f"Error encountered during CSV export: {e}")

    def calc_bpm(self, ir_data_segment):
        """
        Calculates the Beats Per Minute (BPM) from a given segment of IR data
        using peak detection algorithms. The process involves applying filters
        to enhance the accuracy of peak identification.
        """
        if len(ir_data_segment) < SAMPLE_FREQ * 2: # Requires at least 2 seconds of data for calculation.
            return 0.0

        # Applies a band-pass filter to the data, which is essential for isolating
        # the pulse signal's frequency components.
        filtered_data = self.butter_bandpass_filter(
            ir_data_segment, lowcut=0.7, highcut=3.0, fs=SAMPLE_FREQ, order=4
        )
        
        # Determines whether the signal needs to be inverted for peak detection.
        # This decision is based on the observed behavior of the raw IR graph
        # during a pulse (whether it goes down or up).
        # data_to_process = -filtered_data # Option to invert the signal to detect valleys as peaks.
        data_to_process = filtered_data   # Option to use the signal directly, detecting peaks as maxima.

        # Ensures the signal exhibits sufficient variance to enable reliable peak detection.
        if np.std(data_to_process) < 50: # A standard deviation threshold to identify flat or noisy signals.
            # print("Warning: Signal is too flat, potentially indicating noise.")
            return 0.0

        # --- Peak detection using scipy.signal.find_peaks ---
        # `distance`: Specifies the minimum horizontal distance (in samples) between detected peaks.
        #             This parameter is adjusted based on `SAMPLE_FREQ` and the expected BPM range.
        #             For example, a max BPM of 200 (3 beats/sec) implies a minimum distance.
        # `prominence`: Defines the prominence of detected peaks, a crucial factor in
        #               distinguishing true pulse peaks from noise. This is often
        #               calibrated relative to the signal's standard deviation.
        peaks, properties = find_peaks(
            data_to_process, 
            distance=int(SAMPLE_FREQ * 0.35), # Minimum 0.35 seconds between peaks (approx. 171 BPM max).
            prominence=np.std(data_to_process) * 0.4 # Adjustable factor for prominence (0.4 is a common starting point).
        )

        # Print statements for debugging peak detection, if uncommented.
        # print(f"Peaks found: {len(peaks)}, Peak indices: {peaks}")
        # print(f"Prominences: {properties.get('prominences', 'N/A')}")

        if len(peaks) < 2:
            return 0.0 # Requires at least two peaks to compute an interval.

        # Computes the intervals between detected peaks and subsequently calculates
        # the average BPM.
        peak_intervals_samples = np.diff(peaks) # Differences between peak indices, in samples.
        
        # Filters out excessively short or long intervals, which may represent
        # noise or measurement artifacts (e.g., outside 40-180 BPM).
        min_interval_samples = SAMPLE_FREQ * (60 / 180) # Corresponds to 180 BPM.
        max_interval_samples = SAMPLE_FREQ * (60 / 40) # Corresponds to 40 BPM.

        valid_intervals = peak_intervals_samples[
            (peak_intervals_samples >= min_interval_samples) & 
            (peak_intervals_samples <= max_interval_samples)
        ]

        if len(valid_intervals) < 1:
            return 0.0 # Returns 0.0 if no valid intervals are found.

        avg_interval_samples = np.mean(valid_intervals) # Calculates the average of valid intervals.

        # Calculates BPM based on the average interval.
        # BPM = (samples per minute) / (samples per beat).
        bpm = (SAMPLE_FREQ * 60) / avg_interval_samples
        
        # Print statements for debugging BPM calculation, if uncommented.
        # print(f"Peak intervals (samples): {peak_intervals_samples}")
        # print(f"Average interval (samples): {avg_interval_samples}")
        # print(f"Calculated BPM (raw): {bpm}")

        # Applies a reasonable BPM threshold (40-180 BPM) to filter out
        # extreme or erroneous values.
        if 40 <= bpm <= 180:
            return bpm
        else:
            return 0.0


    def closeEvent(self, event):
        """
        Manages the application's closure event, ensuring a proper shutdown
        of the sensor and the sensor reading thread.
        """
        print("Application is shutting down...")
        self.running = False  # Sets the flag to terminate the sensor thread's loop.
        if self.thread.is_alive():
            print("Awaiting sensor thread termination...")
            self.thread.join(timeout=2) # Allows the thread a brief period to terminate cleanly.

        if self.sensor:
            self.sensor.shutdown() # Ensures the MAX30102 sensor is powered down.
            print("MAX30102 sensor has been shut down.")
        event.accept() # Accepts the close event, allowing the application to close.


def main():
    """
    The main function responsible for initializing and executing the application.
    It creates the QApplication instance, instantiates the MainApp window,
    displays it, and initiates the PyQt event loop.
    """
    app = QApplication(sys.argv) # Initializes the PyQt application.
    window = MainApp() # Creates an instance of the MainApp window.
    window.show() # Displays the main application window.
    sys.exit(app.exec_()) # Enters the PyQt main loop, handling events.


if __name__ == "__main__":
    main() # Executes the main function when the script is run directly.