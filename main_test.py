import sys
import csv
import time
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout,
    QWidget, QPushButton, QHBoxLayout, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer # Importar QTimer
import pyqtgraph as pg
from scipy.signal import find_peaks, butter, filtfilt
import threading
from datetime import datetime
import os
# Asegúrate de que esta importación sea correcta para tu configuración
# Podrías necesitar instalar la biblioteca de tu sensor (ej. max30102_sensor)
try:
    from sensor.max30102 import MAX30102
except ImportError:
    print("Error: No se pudo importar la clase MAX30102 del módulo 'sensor'.")
    print("Asegúrate de que el archivo max30102.py esté en una carpeta 'sensor' en el mismo directorio que este script, o que la biblioteca esté instalada correctamente.")
    sys.exit(1)


SAMPLE_FREQ = 100 # Frecuencia de muestreo del sensor en Hz (ajusta si es diferente)
BUFFER_SIZE_BPM = SAMPLE_FREQ * 4 # Mantener al menos 4 segundos de datos para el cálculo de BPM
BUFFER_SIZE_PLOT = 100 # Número de muestras a mostrar en la gráfica
MA_SIZE = 4 # Tamaño de la media móvil para suavizado (en desuso con filtro, pero se mantiene si lo necesitas)


class DataEmitter(QObject):
    """
    QObject subclass used to emit new IR data batches via a PyQt signal.
    This allows the sensor reading thread to communicate data to the GUI thread safely.
    """
    new_data = pyqtSignal(list)  # Signal to send a batch (list) of IR data points
    new_bpm_status = pyqtSignal(float, str) # Señal para actualizar BPM y estado

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
        self.buffer_size = BUFFER_SIZE_PLOT  # Number of samples to keep in buffer for plotting
        self.data = np.zeros(self.buffer_size, dtype=int)  # Circular buffer for IR values
        self.ptr = 0  # Current index in the buffer

    def update_plot(self, new_vals):
        """
        Update the plot with a new batch of IR data.
        Maintains a circular buffer of the latest samples.
        """
        n = len(new_vals)
        if n == 0: # No hay nuevos valores
            return

        if n > self.buffer_size:
            # If new data is bigger than buffer, keep only the most recent part
            new_vals = new_vals[-self.buffer_size:]
            n = self.buffer_size

        if self.ptr + n <= self.buffer_size: # Usar <= para evitar IndexError en el límite
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
        self.label_status = QLabel("Status: Esperando dedo...", alignment=Qt.AlignCenter)
        self.label_status.setStyleSheet("font-size: 14pt; color: #666;")

        # Button to start/stop the measurement
        self.btn_toggle = QPushButton("Iniciar Medición")
        self.btn_toggle.setStyleSheet("padding: 10px; font-size: 12pt;")
        self.btn_toggle.clicked.connect(self.toggle_measurement)

        # Button to export collected data to CSV
        self.btn_export = QPushButton("Exportar CSV")
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
        try:
            self.sensor = MAX30102()
            print("Sensor MAX30102 inicializado correctamente.")
        except Exception as e:
            print(f"Error al inicializar el sensor MAX30102: {e}")
            self.label_status.setText("Error: Sensor no inicializado.")
            self.btn_toggle.setEnabled(False) # Deshabilitar botón si el sensor falla
            self.sensor = None # Asegurarse de que el sensor sea None si falla

        # List to store IR data for BPM calculation and CSV export
        self.ir_data = []

        # List to store data rows for CSV export
        self.csv_data = []

        # Flag indicating whether measurement is active
        self.measuring = False

        # Create data emitter for thread-safe signal emission
        self.data_emitter = DataEmitter()
        self.data_emitter.new_data.connect(self.update_plot)
        self.data_emitter.new_bpm_status.connect(self.update_bpm_status_labels) # Conectar nueva señal

        self.running = True  # Flag to control the sensor thread
        
        # Start the sensor reading thread
        if self.sensor: # Solo iniciar el hilo si el sensor se inicializó correctamente
            self.thread = threading.Thread(target=self.read_sensor)
            self.thread.daemon = True
            self.thread.start()

        self.bpm_history = []  # For BPM smoothing
        self.IR_THRESHOLD = 50000  # Umbral para detectar si hay dedo. Ajusta según tus pruebas.
                                  # Un valor típico para 'sin dedo' es cercano a 0, con dedo > 10000.
                                  # Un valor demasiado bajo puede causar lecturas falsas.
                                  # Un valor muy alto puede no detectar el dedo.

        # QTimer para actualizar el estado del BPM y la interfaz periódicamente
        self.bpm_timer = QTimer(self)
        self.bpm_timer.setInterval(1000) # Actualizar cada 1000 ms (1 segundo)
        self.bpm_timer.timeout.connect(self.calculate_and_update_bpm)
        self.bpm_timer.start()


    def toggle_measurement(self):
        """
        Start or stop measuring pulse.
        Clears data when starting; exports CSV when stopping.
        """
        if not self.sensor: # No permitir iniciar si el sensor no está inicializado
            return

        self.measuring = not self.measuring
        if self.measuring:
            self.ir_data.clear()
            self.csv_data.clear()
            self.bpm_history.clear()
            self.label_bpm.setText("BPM: 0")
            self.label_status.setText("Status: Midiendo...")
            self.btn_toggle.setText("Detener Medición")
        else:
            self.btn_toggle.setText("Iniciar Medición")
            self.export_csv(auto=True)
            self.label_status.setText("Status: Medición detenida")
            self.label_bpm.setText("BPM: 0")


    def read_sensor(self):
        """
        Reads data from the MAX30102 sensor in a separate thread.
        Emits data batches for plotting and accumulates data for BPM calculation.
        """
        batch_size = 10 # Número de muestras a enviar al plot en cada actualización
        batch = []
        
        # Ajustar el delay basado en la frecuencia de muestreo si el sensor lo requiere
        # Si el sensor ya tiene su propio FIFO buffer y lee a una tasa fija, este sleep es para no saturar
        # Si no, ajusta SAMPLE_FREQ y el sleep para obtener la tasa deseada
        read_delay = 1.0 / SAMPLE_FREQ # Intentar leer a la frecuencia de muestreo deseada

        while self.running:
            try:
                # Intenta leer una muestra del sensor
                # Asegúrate de que tu método read_fifo() devuelva un diccionario con 'red' e 'ir'
                # o ajusta según la implementación real de tu biblioteca del sensor.
                fifo_data = self.sensor.read_fifo()
                
                if fifo_data: # Asegúrate de que se leyeron datos válidos
                    # red = fifo_data['red'] # El código no usa el valor RED actualmente
                    ir = fifo_data['ir']

                    # Añadir al lote para el plot
                    batch.append(ir)

                    if self.measuring:
                        # Acumular datos para el cálculo de BPM y CSV
                        self.ir_data.append(ir)
                        # Mantener el buffer de ir_data a un tamaño manejable para BPM
                        if len(self.ir_data) > BUFFER_SIZE_BPM + 50: # Mantener un poco más del necesario
                             self.ir_data = self.ir_data[-BUFFER_SIZE_BPM - 50:]


                    # Emitir el lote para el plot cuando sea suficientemente grande
                    if len(batch) >= batch_size:
                        self.data_emitter.new_data.emit(batch)
                        batch = [] # Limpiar el lote después de emitirlo

                time.sleep(read_delay) # Controlar la frecuencia de lectura

            except Exception as e:
                print(f"Error en el hilo de lectura del sensor: {e}")
                # Podrías añadir un mensaje de error en la GUI si es crítico
                self.label_status.setText("Error: Fallo de sensor.")
                time.sleep(2) # Esperar un poco antes de reintentar para no saturar


    def update_bpm_status_labels(self, bpm, status):
        """
        Slot to update the BPM and status labels in the GUI.
        Called via a signal from the sensor reading thread.
        """
        self.label_bpm.setText(f"BPM: {bpm:.1f}")
        self.label_status.setText(f"Status: {status}")

    def calculate_and_update_bpm(self):
        """
        Calculates BPM and updates status labels periodically.
        This is called by a QTimer to ensure GUI updates happen in the main thread.
        """
        if self.measuring:
            current_ir_data = np.array(self.ir_data)

            bpm = 0
            status = "Esperando datos..."

            # Comprobar si hay suficientes datos para un cálculo significativo
            if len(current_ir_data) < BUFFER_SIZE_BPM:
                status = "Recolectando datos..."
            elif np.mean(current_ir_data[-SAMPLE_FREQ:]) < self.IR_THRESHOLD: # Comprobar último segundo
                status = "No hay dedo detectado"
                self.bpm_history.clear() # Limpiar historial si no hay dedo
            else:
                bpm = self.calc_bpm(current_ir_data)
                status = self.classify_bpm(bpm)

                # Suavizar BPM usando un promedio móvil de los últimos valores
                if bpm > 0: # Solo añadir BPMs válidos al historial
                    self.bpm_history.append(bpm)
                    if len(self.bpm_history) > 5: # Mantener los últimos 5 valores
                        self.bpm_history.pop(0)
                    bpm = sum(self.bpm_history) / len(self.bpm_history)
                else:
                    self.bpm_history.clear() # Si BPM es 0, reiniciar historial de suavizado


            # Emitir la señal para actualizar las etiquetas de la GUI
            # Esto se hace desde el hilo principal (QTimer) y es seguro.
            self.data_emitter.new_bpm_status.emit(bpm, status)

            # Guardar en CSV solo si hay datos válidos y se está midiendo
            if self.measuring and len(current_ir_data) > 0:
                timestamp = time.time()
                readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                # Aquí se está guardando solo el último valor de IR, considera si quieres un promedio o el lote.
                # Para mayor precisión, podrías guardar el lote completo de 'ir' que lleva el `read_sensor`
                # o el valor promedio del lote si 'ir' es un valor individual.
                # Por ahora, se guarda el último valor de IR del buffer para CSV.
                self.csv_data.append((timestamp, readable_time, current_ir_data[-1], f"{bpm:.1f}", status))


    def butter_lowpass_filter(self, data, cutoff=2.5, fs=SAMPLE_FREQ, order=2):
        """
        Applies a Butterworth low-pass filter to the IR data.
        :param data: list or numpy array of IR data
        :param cutoff: cutoff frequency in Hz
        :param fs: sampling frequency in Hz
        :param order: filter order
        :return: filtered data as numpy array
        """
        nyq = 0.5 * fs  # Nyquist frequency
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def butter_bandpass_filter(self, data, lowcut=0.7, highcut=3.0, fs=SAMPLE_FREQ, order=2):
        """
        Applies a Butterworth band-pass filter to the IR data.
        A band-pass filter is often better for pulse detection as it isolates
        the frequency range of heartbeats (e.g., 0.7 Hz to 3.0 Hz, 42-180 BPM).
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    
    def classify_bpm(self, bpm):
        """
        Classify the heart rate status based on BPM value.
        :param bpm: beats per minute
        :return: string status ("Bradycardia", "Tachycardia", or "Normal")
        """
        if bpm == 0: # Cuando el BPM es 0, es mejor indicar que no hay lectura
            return "No hay lectura"
        elif bpm < 60:
            return "Bradicardia"
        elif bpm > 90:
            return "Taquicardia"
        else:
            return "Normal"

    def export_csv(self, auto=False):
        """
        Export the collected measurement data to a CSV file.
        If auto=True, saves to a default folder with timestamped filename.
        Otherwise, opens a save dialog.
        """
        if not self.csv_data:
            print("No hay datos para exportar.")
            return

        if auto:
            folder = os.path.expanduser('~/max30102_exports')
            os.makedirs(folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = os.path.join(folder, f"max30102_data_{timestamp}.csv")
        else:
            file_name, _ = QFileDialog.getSaveFileName(self, "Guardar CSV", "", "Archivos CSV (*.csv)")
            if not file_name:
                return

        try:
            with open(file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp Unix', 'Tiempo Legible', 'Valor IR', 'BPM', 'Estado'])
                writer.writerows(self.csv_data)
            print("Datos exportados a:", file_name)
        except Exception as e:
            print(f"Error al exportar CSV: {e}")

    def calc_bpm(self, ir_data):
        """
        Calculates BPM from IR data using peak detection.
        Applies filtering for better peak detection.
        """
        if len(ir_data) < SAMPLE_FREQ * 2: # Necesitamos al menos 2 segundos de datos
            return 0

        # Convertir a numpy array y aplicar filtro de paso banda
        # El filtro de paso banda es crucial para aislar las frecuencias del pulso
        filtered_data = self.butter_bandpass_filter(ir_data, lowcut=0.7, highcut=3.0, fs=SAMPLE_FREQ, order=4)
        
        # Invertir la señal si las pulsaciones son valles (común en muchos sensores IR)
        # o mantenerla si son picos. Puedes necesitar experimentar con esto.
        # Si la señal del sensor baja durante un pulso, invierte:
        # data_to_process = -filtered_data
        # Si la señal del sensor sube durante un pulso, no inviertas:
        data_to_process = filtered_data

        # Encuentra picos prominentes en la señal filtrada.
        # Ajusta `distance` (distancia mínima entre picos) y `prominence` (prominencia de los picos)
        # en base a tus datos. SAMPLE_FREQ * 0.5 equivale a 2 picos por segundo (120 BPM) como mínimo.
        # `prominence` asegura que solo se detecten picos significativos.
        peaks, properties = find_peaks(data_to_process, 
                                       distance=int(SAMPLE_FREQ * 0.4), # Distancia mínima para 150 BPM (60/150 = 0.4s)
                                       prominence=np.std(data_to_process) * 0.5) # Prominencia para evitar ruido

        if len(peaks) < 2:
            return 0 # Necesitamos al menos dos picos para calcular un intervalo

        # Calcular los intervalos entre picos y luego el BPM promedio
        peak_intervals = np.diff(peaks) # Diferencias entre los índices de los picos
        avg_interval_samples = np.mean(peak_intervals)

        # BPM = (muestras por minuto) / (muestras por latido)
        # BPM = (SAMPLE_FREQ * 60) / avg_interval_samples
        bpm = (SAMPLE_FREQ * 60) / avg_interval_samples
        
        # Añadir un umbral de BPM razonable para evitar valores erróneos
        if 40 <= bpm <= 180: # Rango de BPM esperado para humanos
            return bpm
        else:
            return 0


    def closeEvent(self, event):
        """
        Handles application close event.
        Properly shuts down the sensor and the reading thread.
        """
        print("Cerrando aplicación...")
        self.running = False  # Set flag to stop thread loop
        if self.thread.is_alive():
            self.thread.join(timeout=2) # Give the thread a chance to finish

        if self.sensor:
            self.sensor.shutdown()
            print("Sensor MAX30102 apagado.")
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