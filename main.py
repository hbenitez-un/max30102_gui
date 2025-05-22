import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import threading
import time
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt

# Import the sensor driver
from sensor.max30102 import MAX30102


class PulsePlot(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.data = [0] * 100
        self.line, = self.ax.plot(self.data, 'r-')
        self.ax.set_ylim(0, 200000)

    def update_plot(self, new_val):
        self.data = self.data[1:] + [new_val]
        self.line.set_ydata(self.data)
        self.ax.set_ylim(min(self.data) - 1000, max(self.data) + 1000)  # autoajuste
        self.draw()
        self.ax.set_title("MAX30102 Pulse Monitor")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("IR LED Value")
        self.ax.grid(True)



class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAX30102 Pulse Monitor")
        self.resize(600, 400)

        self.plot = PulsePlot(self)
        self.label_bpm = QLabel("BPM: 0")
        self.label_status = QLabel("Status: Normal")

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

                # Limita el tamaño del buffer
                if len(ir_data) > 300:
                    ir_data = ir_data[-300:]

                if len(ir_data) >= 200:
                    bpm = self.calc_bpm(ir_data)
                    self.label_bpm.setText(f"BPM: {bpm:.1f}")
                    self.label_status.setText(f"Status: {self.classify_bpm(bpm)}")

                time.sleep(0.05)  # más rápido y más responsivo que 1s
            except Exception as e:
                print(f"Error reading sensor: {e}")
                time.sleep(1)

    def butter_lowpass_filter(data, cutoff=2.5, fs=100, order=2):
            nyq = 0.5 * fs  # Frecuencia de Nyquist
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
        ''' Override closeEvent to handle cleanup before closing the window'''
        self.sensor.shutdown()
        event.accept()




def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
