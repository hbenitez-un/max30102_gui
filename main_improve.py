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
from scipy.signal import find_peaks # Solo usaremos find_peaks
import threading
from datetime import datetime
import os
from sensor.max30102 import MAX30102


class DataEmitter(QObject):
    """
    Clase QObject para emitir lotes de datos IR a través de una señal PyQt.
    Permite que el hilo de lectura del sensor se comunique con el hilo de la GUI de forma segura.
    """
    new_data = pyqtSignal(list)  # Señal para enviar un lote (lista) de puntos de datos IR


class PulsePlot(pg.PlotWidget):
    """
    Widget de gráfica personalizado para mostrar datos del sensor IR en tiempo real.
    Utiliza un búfer circular para mantener una cantidad fija de puntos de datos.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setBackground('w')  # Fondo blanco
        self.setTitle("Monitor de Pulso MAX30102", color='k', size='16pt')
        self.setLabel('left', "Valor IR", **{'color': 'black', 'font-size': '12pt'})
        self.setLabel('bottom', "Muestra", **{'color': 'black', 'font-size': '12pt'})
        self.showGrid(x=True, y=True)

        self.curve = self.plot(pen=pg.mkPen(color='r', width=2))  # Línea roja para los datos
        self.buffer_size = 100  # Número de muestras a mantener en el búfer
        self.data = np.zeros(self.buffer_size, dtype=int)  # Búfer circular para valores IR
        self.ptr = 0  # Índice actual en el búfer

    def update_plot(self, new_vals):
        """
        Actualiza la gráfica con un nuevo lote de datos IR.
        Mantiene un búfer circular de las últimas muestras.
        """
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
    """
    Ventana principal de la aplicación que contiene la gráfica de pulso, etiquetas y botones.
    Gestiona la lectura del sensor en un hilo separado y el cálculo de BPM.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monitor de Pulso MAX30102")
        self.resize(700, 500)

        self.plot = PulsePlot(self)

        self.label_bpm = QLabel("BPM: 0", alignment=Qt.AlignCenter)
        self.label_bpm.setStyleSheet("font-size: 24pt; font-weight: bold; color: #333;")

        self.label_status = QLabel("Estado: Sin Datos", alignment=Qt.AlignCenter) # Estado más simple
        self.label_status.setStyleSheet("font-size: 14pt; color: #666;")

        self.btn_toggle = QPushButton("Iniciar Medición")
        self.btn_toggle.setStyleSheet("padding: 10px; font-size: 12pt;")
        self.btn_toggle.clicked.connect(self.toggle_measurement)

        self.btn_export = QPushButton("Exportar CSV")
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

        self.ir_data = [] # Lista para almacenar datos IR
        self.csv_data = [] # Lista para almacenar datos para exportar a CSV

        self.measuring = False
        self.data_emitter = DataEmitter()
        self.data_emitter.new_data.connect(self.update_plot)
        self.running = True

        self.thread = threading.Thread(target=self.read_sensor)
        self.thread.daemon = True
        self.thread.start()

        self.SAMPLING_RATE = 100 # Frecuencia de muestreo en Hz
        self.IR_THRESHOLD = 50000 # Umbral para detectar si hay un dedo


    def toggle_measurement(self):
        """
        Inicia o detiene la medición del pulso.
        """
        self.measuring = not self.measuring
        if self.measuring:
            self.ir_data.clear()
            self.csv_data.clear()
            self.label_status.setText("Estado: Midiendo...")
            self.btn_toggle.setText("Detener Medición")
        else:
            self.btn_toggle.setText("Iniciar Medición")
            self.export_csv(auto=True)

    def read_sensor(self):
        batch_size = 10
        batch = []

        while self.running:
            try:
                red, ir = self.sensor.read_fifo()
                if ir is not None:
                    batch.append(ir)

                    if len(batch) >= batch_size:
                        self.data_emitter.new_data.emit(batch)
                        batch.clear()

                    if self.measuring:
                        timestamp = time.time()
                        readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

                        self.ir_data.append(ir)
                        # Mantener las últimas 5 segundos de datos para cálculo de BPM
                        if len(self.ir_data) > self.SAMPLING_RATE * 5:
                            self.ir_data = self.ir_data[-self.SAMPLING_RATE * 5:]

                        bpm = 0
                        status = "Sin Dedo"

                        # Verificar si hay un dedo usando el umbral IR
                        if max(self.ir_data[-min(batch_size, len(self.ir_data)):]) < self.IR_THRESHOLD:
                            status = "Sin Dedo"
                            bpm = 0
                        elif len(self.ir_data) >= self.SAMPLING_RATE * 3: # Necesitamos al menos 3 segundos de datos
                            bpm = self.calc_bpm(self.ir_data)
                            status = "Normal" if 40 < bpm < 100 else ("Bradicardia" if bpm <= 40 else "Taquicardia")


                        self.label_bpm.setText(f"BPM: {bpm:.1f}")
                        self.label_status.setText(f"Estado: {status}")

                        self.csv_data.append((timestamp, readable_time, ir, f"{bpm:.1f}", status))

                time.sleep(0.01) # Pequeño retardo para no saturar la CPU

            except Exception as e:
                print(f"Error leyendo sensor: {e}")
                time.sleep(1) # Esperar un segundo si hay un error

    def update_plot(self, new_vals):
        """
        Slot para actualizar el widget de la gráfica con nuevos datos IR.
        """
        self.plot.update_plot(new_vals)

    def calc_bpm(self, ir_data, fs=100):
        """
        Calcula el BPM a partir de los datos IR usando una detección de picos básica.
        """
        data = np.array(ir_data)

        # En esta versión simplificada, no aplicamos filtros complejos.
        # Solo detrending básico para eliminar la deriva de la línea base.
        detrended_data = data - np.mean(data)

        # Buscar picos en los datos.
        # El parámetro 'distance' ayuda a asegurar que no se detecten demasiados picos muy juntos.
        # Un valor de 0.6 segundos entre picos (60 muestras a 100Hz) es razonable para un pulso humano.
        peaks, _ = find_peaks(detrended_data, distance=int(fs * 0.6))

        if len(peaks) < 2:
            return 0  # No hay suficientes picos para calcular el BPM

        # Calcular los intervalos entre picos y luego el BPM
        rr_intervals = np.diff(peaks) / fs
        bpm = 60 / np.mean(rr_intervals)
        return bpm

    def export_csv(self, auto=False):
        """
        Exporta los datos de medición recolectados a un archivo CSV.
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
                writer.writerow(['Timestamp', 'Readable Time', 'IR_Value', 'BPM', 'Status'])
                writer.writerows(self.csv_data)
            print("Datos exportados a:", file_name)
        except IOError as e:
            print(f"No se pudo guardar el archivo: {e}")

    def closeEvent(self, event):
        """
        Maneja el evento de cierre de la aplicación.
        Apaga correctamente el sensor.
        """
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        self.sensor.shutdown()
        event.accept()


def main():
    """
    Función principal para crear y ejecutar la aplicación.
    """
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()