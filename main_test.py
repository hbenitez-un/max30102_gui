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

# Asegúrate de que esta importación sea correcta para tu configuración.
# Si tu librería del sensor devuelve los datos como una tupla (rojo, IR),
# este import debe ser de la clase MAX30102 que se comporta así.
# Si devuelve un diccionario, el código se adaptará, pero la tupla es más común.
try:
    # Si tu clase MAX30102 está dentro de un paquete 'sensor'
    from sensor.max30102 import MAX30102
except ImportError:
    # Si tu clase MAX30102 está en un archivo llamado max30102.py en el mismo nivel
    # o si es directamente de una librería instalada como 'max30102_sensor'
    try:
        from max30102 import MAX30102 # Intenta importar directamente el archivo
    except ImportError:
        try:
            from max30102_sensor import MAX30102 # Intenta importar si es una librería
        except ImportError:
            print("Error: No se pudo importar la clase MAX30102.")
            print("Asegúrate de que la librería del sensor esté instalada o que el archivo 'max30102.py' esté accesible.")
            sys.exit(1)


# --- Constantes de Configuración ---
# Ajusta estas constantes según tu sensor y tus necesidades.
# SAMPLE_FREQ: Frecuencia de muestreo real de tu sensor MAX30102.
# Este valor es CRÍTICO para el cálculo de BPM. Consulta la documentación
# de tu módulo o de la librería del sensor. Valores comunes son 25, 50, 100 Hz.
SAMPLE_FREQ = 100 # Hz (ej. 100 Hz es común para el MAX30102)

# BUFFER_SIZE_PLOT: Cuántas muestras se muestran en la gráfica en tiempo real.
BUFFER_SIZE_PLOT = SAMPLE_FREQ * 5 # Mostrar 5 segundos de datos en la gráfica

# BUFFER_SIZE_BPM: Cuántas muestras se usan para el cálculo de BPM.
# Unos 4-6 segundos de datos son buenos para una medición estable.
BUFFER_SIZE_BPM = SAMPLE_FREQ * 4 # Usar 4 segundos de datos para BPM

# IR_THRESHOLD: Umbral para detectar si hay un dedo.
# Ajusta este valor probando tu sensor:
# - Con el dedo puesto, el valor IR debe ser ALTO (ej. 50000 - 200000).
# - Sin el dedo, el valor IR debe ser BAJO (ej. 0 - 5000).
# El umbral debe estar entre estos dos rangos.
IR_THRESHOLD = 15000 # Valor IR mínimo para considerar que hay un dedo


class DataEmitter(QObject):
    """
    Subclase QObject usada para emitir lotes de datos IR nuevos vía una señal PyQt.
    Esto permite que el hilo de lectura del sensor comunique datos al hilo de la GUI de forma segura.
    """
    new_data = pyqtSignal(list)  # Señal para enviar un lote (lista) de puntos de datos IR
    new_bpm_status = pyqtSignal(float, str) # Señal para actualizar BPM y estado

class PulsePlot(pg.PlotWidget):
    """
    Widget de gráfica personalizado para mostrar datos del sensor IR en tiempo real.
    Usa un buffer circular para mantener una cantidad fija de puntos de datos.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setBackground('w')  # Fondo blanco
        self.setTitle("Monitor de Pulso MAX30102", color='k', size='16pt')
        self.setLabel('left', "Valor IR", **{'color': 'black', 'font-size': '12pt'})
        self.setLabel('bottom', "Muestra", **{'color': 'black', 'font-size': '12pt'})
        self.showGrid(x=True, y=True)

        # Curva principal para los datos IR crudos
        self.curve = self.plot(pen=pg.mkPen(color='r', width=2), name="IR Crudo") 
        # (Opcional) Curva para visualizar la señal filtrada (útil para depuración)
        # self.curve_filtered = self.plot(pen=pg.mkPen(color='b', width=1), name="IR Filtrado") 

        self.buffer_size = BUFFER_SIZE_PLOT  # Número de muestras a mantener en el buffer para graficar
        self.data = np.zeros(self.buffer_size, dtype=int)  # Buffer circular para valores IR
        self.ptr = 0  # Índice actual en el buffer

    def update_plot(self, new_vals):
        """
        Actualiza la gráfica con un nuevo lote de datos IR.
        Mantiene un buffer circular de las últimas muestras.
        """
        n = len(new_vals)
        if n == 0:
            return

        # Si los nuevos datos son más grandes que el buffer, solo se toma la parte más reciente
        if n > self.buffer_size:
            new_vals = new_vals[-self.buffer_size:]
            n = self.buffer_size

        if self.ptr + n <= self.buffer_size:
            # Si los nuevos datos caben sin envolver, insertar directamente
            self.data[self.ptr:self.ptr + n] = new_vals
        else:
            # Dividir los datos para que se envuelvan alrededor del final del buffer
            till_end = self.buffer_size - self.ptr
            self.data[self.ptr:] = new_vals[:till_end]
            self.data[:n - till_end] = new_vals[till_end:]
        self.ptr = (self.ptr + n) % self.buffer_size

        # Reorganizar los datos para que la gráfica muestre una forma de onda continua
        if self.ptr == 0:
            display_data = self.data
        else:
            display_data = np.concatenate((self.data[self.ptr:], self.data[:self.ptr]))

        self.curve.setData(display_data)

        # (Opcional) Si descomentaste self.curve_filtered, aquí podrías actualizarlo.
        # Esto requeriría una forma de pasar datos filtrados al plot.
        # self.curve_filtered.setData(display_filtered_data)


class MainApp(QMainWindow):
    """
    Ventana principal de la aplicación que contiene la gráfica de pulso, etiquetas y botones.
    Maneja la lectura del sensor en un hilo separado y el cálculo de BPM.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monitor de Pulso MAX30102")
        self.resize(700, 500)

        # Crear y añadir el widget de la gráfica de pulso
        self.plot = PulsePlot(self)

        # Etiqueta para mostrar el BPM (pulsaciones por minuto)
        self.label_bpm = QLabel("BPM: 0", alignment=Qt.AlignCenter)
        self.label_bpm.setStyleSheet("font-size: 24pt; font-weight: bold; color: #333;")

        # Etiqueta para mostrar el estado de la frecuencia cardíaca
        self.label_status = QLabel("Estado: Iniciando sensor...", alignment=Qt.AlignCenter)
        self.label_status.setStyleSheet("font-size: 14pt; color: #666;")

        # Botón para iniciar/detener la medición
        self.btn_toggle = QPushButton("Iniciar Medición")
        self.btn_toggle.setStyleSheet("padding: 10px; font-size: 12pt;")
        self.btn_toggle.clicked.connect(self.toggle_measurement)

        # Botón para exportar los datos recolectados a CSV
        self.btn_export = QPushButton("Exportar CSV")
        self.btn_export.setStyleSheet("padding: 10px; font-size: 12pt;")
        self.btn_export.clicked.connect(self.export_csv)

        # Layout para los botones uno al lado del otro
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_toggle)
        button_layout.addWidget(self.btn_export)

        # Layout vertical principal
        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        layout.addWidget(self.label_bpm)
        layout.addWidget(self.label_status)
        layout.addLayout(button_layout)

        # Establecer el widget central con el layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Inicializar la instancia del sensor MAX30102
        self.sensor = None # Inicializar a None
        try:
            self.sensor = MAX30102()
            print("Sensor MAX30102 inicializado correctamente.")
            self.label_status.setText("Estado: Esperando dedo...")
        except Exception as e:
            print(f"Error al inicializar el sensor MAX30102: {e}")
            self.label_status.setText("Error: Sensor no inicializado. Verifique conexiones y librerías.")
            self.btn_toggle.setEnabled(False) # Deshabilitar botón si el sensor falla

        # Lista para almacenar datos IR para el cálculo de BPM y exportación CSV
        self.ir_data = []
        # Lista para almacenar filas de datos para exportación CSV
        self.csv_data = []
        # Bandera que indica si la medición está activa
        self.measuring = False

        # Crear emisor de datos para emisión de señales segura entre hilos
        self.data_emitter = DataEmitter()
        # Conectar la señal de nuevos datos IR a la función de actualización de la gráfica
        self.data_emitter.new_data.connect(self.plot.update_plot) # CORRECCIÓN: Conectar a self.plot.update_plot
        # Conectar la señal de BPM y estado a la función de actualización de etiquetas
        self.data_emitter.new_bpm_status.connect(self.update_bpm_status_labels)

        self.running = True  # Bandera para controlar el bucle del hilo del sensor
        
        # Iniciar el hilo de lectura del sensor solo si el sensor se inicializó correctamente
        if self.sensor:
            self.thread = threading.Thread(target=self.read_sensor)
            self.thread.daemon = True # El hilo se detendrá cuando la aplicación principal se cierre
            self.thread.start()

        self.bpm_history = []  # Para suavizar el BPM
        
        # QTimer para actualizar el estado del BPM y la interfaz periódicamente
        # Esto asegura que las actualizaciones de la GUI se hagan en el hilo principal
        self.bpm_timer = QTimer(self)
        self.bpm_timer.setInterval(500) # Actualizar cada 500 ms (0.5 segundos)
        self.bpm_timer.timeout.connect(self.calculate_and_update_bpm)
        self.bpm_timer.start()


    def toggle_measurement(self):
        """
        Inicia o detiene la medición del pulso.
        Limpia los datos al iniciar; exporta CSV al detener.
        """
        if not self.sensor: # No permitir iniciar si el sensor no está inicializado
            return

        self.measuring = not self.measuring
        if self.measuring:
            self.ir_data.clear()
            self.csv_data.clear()
            self.bpm_history.clear()
            self.label_bpm.setText("BPM: 0")
            self.label_status.setText("Estado: Midiendo...")
            self.btn_toggle.setText("Detener Medición")
        else:
            self.btn_toggle.setText("Iniciar Medición")
            self.export_csv(auto=True) # Exportar automáticamente al detener
            self.label_status.setText("Estado: Medición detenida")
            self.label_bpm.setText("BPM: 0")


    def read_sensor(self):
        """
        Lee datos del sensor MAX30102 en un hilo separado.
        Emite lotes de datos para la gráfica y acumula datos para el cálculo de BPM.
        """
        batch_size = 10 # Número de muestras a enviar al plot en cada actualización
        batch = []
        
        # Calcula el retraso necesario para aproximar la frecuencia de muestreo deseada
        read_delay = 1.0 / SAMPLE_FREQ # ej. 1.0 / 100 Hz = 0.01 segundos

        while self.running:
            try:
                # Asegúrate de que el método read_fifo() de tu clase MAX30102
                # devuelve una tupla (red_value, ir_value) o un diccionario {'red': ..., 'ir': ...}.
                # El código se adapta a ambos, pero el error anterior sugería una tupla.
                fifo_data = self.sensor.read_fifo()
                
                if fifo_data is None:
                    # Si no hay datos (ej. FIFO vacío o error de lectura interna del sensor)
                    time.sleep(read_delay / 2) # Esperar un poco antes de reintentar
                    continue # Saltar al siguiente ciclo del bucle

                # --- Adaptación para el formato de datos de read_fifo() ---
                ir = 0
                if isinstance(fifo_data, tuple) and len(fifo_data) >= 2:
                    # Asumiendo que el formato es (red_value, ir_value)
                    ir = fifo_data[1] # El valor IR es el segundo elemento
                elif isinstance(fifo_data, dict) and 'ir' in fifo_data:
                    # Asumiendo que el formato es {'red': ..., 'ir': ...}
                    ir = fifo_data['ir']
                else:
                    print(f"Advertencia: read_fifo() devolvió un formato inesperado: {type(fifo_data)} - {fifo_data}. Saltando ciclo.")
                    time.sleep(read_delay / 2)
                    continue # Saltar al siguiente ciclo del bucle

                # Filtrar valores IR extremadamente bajos que pueden ser ruido
                if ir < 100: # Un umbral bajo para ignorar lecturas de ruido cero
                    ir = 0 # O puedes optar por simplemente 'continue' para no añadir estos datos

                # Añadir al lote para el plot
                batch.append(ir)

                if self.measuring:
                    # Acumular datos para el cálculo de BPM y CSV
                    self.ir_data.append(ir)
                    # Mantener el buffer de ir_data a un tamaño manejable para BPM
                    if len(self.ir_data) > BUFFER_SIZE_BPM + SAMPLE_FREQ * 2: # Mantener un poco más del necesario
                         self.ir_data = self.ir_data[-BUFFER_SIZE_BPM - SAMPLE_FREQ * 2:]

                # Emitir el lote para el plot cuando sea suficientemente grande
                if len(batch) >= batch_size:
                    self.data_emitter.new_data.emit(batch)
                    batch = [] # Limpiar el lote después de emitirlo

                time.sleep(read_delay) # Controlar la frecuencia de lectura

            except Exception as e:
                print(f"Error en el hilo de lectura del sensor: {e}")
                # Si el sensor se desconecta o falla, esto lo capturará
                # Podrías actualizar el estado de la GUI con un error más específico aquí si lo deseas
                self.label_status.setText("Error: Fallo de sensor. Reintentando...")
                time.sleep(2) # Esperar un poco antes de reintentar para no saturar


    def update_bpm_status_labels(self, bpm, status):
        """
        Slot para actualizar las etiquetas de BPM y estado en la GUI.
        Llamado vía una señal del QTimer (que se ejecuta en el hilo principal).
        """
        self.label_bpm.setText(f"BPM: {bpm:.1f}")
        self.label_status.setText(f"Estado: {status}")

    def calculate_and_update_bpm(self):
        """
        Calcula el BPM y actualiza las etiquetas de estado periódicamente.
        Esto es llamado por un QTimer para asegurar que las actualizaciones de la GUI
        ocurran en el hilo principal de forma segura.
        """
        if not self.measuring:
            # Si no estamos midiendo, no calculamos ni actualizamos
            self.data_emitter.new_bpm_status.emit(0.0, "Medición detenida")
            return

        current_ir_data = np.array(self.ir_data, dtype=float) # Convertir a float para filtros

        bpm = 0.0
        status = "Recolectando datos..."

        # 1. Comprobar la presencia del dedo
        # Se verifica que haya suficientes datos y que el promedio de IR reciente
        # esté por encima del umbral.
        if len(current_ir_data) < SAMPLE_FREQ: # Necesitamos al menos 1 segundo de datos
            status = "Recolectando datos..."
        # --- CORRECCIÓN AQUÍ ---
        elif np.mean(current_ir_data[-SAMPLE_FREQ:]) < IR_THRESHOLD: # Usa IR_THRESHOLD directamente
        # --- FIN CORRECCIÓN ---
            status = "No hay dedo detectado"
            self.bpm_history.clear() # Limpiar historial si no hay dedo
        else:
            # 2. Calcular BPM si hay suficientes datos y se detecta un dedo
            if len(current_ir_data) >= BUFFER_SIZE_BPM:
                # Se toma un segmento de datos para el cálculo (los más recientes)
                segment_to_process = current_ir_data[-BUFFER_SIZE_BPM:]
                
                bpm = self.calc_bpm(segment_to_process)
                
                # 3. Suavizar BPM si es válido y clasificar el estado
                if 40 <= bpm <= 180: # Solo suavizar y clasificar BPMs razonables
                    self.bpm_history.append(bpm)
                    if len(self.bpm_history) > 10: # Mantener los últimos 10 valores para suavizado
                        self.bpm_history.pop(0)
                    bpm = sum(self.bpm_history) / len(self.bpm_history)
                    status = self.classify_bpm(bpm)
                else:
                    self.bpm_history.clear() # Si el BPM calculado es inválido, reiniciar historial
                    status = "Analizando señal..." # O "BPM fuera de rango"
                    bpm = 0.0 # Restablecer BPM a 0 si es inválido
            else:
                status = "Recolectando datos..."

        # Emitir la señal para actualizar las etiquetas de la GUI (seguro por QTimer)
        self.data_emitter.new_bpm_status.emit(bpm, status)

        # 4. Guardar en CSV solo si hay datos válidos y se está midiendo
        if self.measuring and len(self.ir_data) > 0:
            timestamp = time.time()
            readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            # Guarda el último valor IR leído para el CSV
            self.csv_data.append((timestamp, readable_time, self.ir_data[-1], f"{bpm:.1f}", status))

    def butter_lowpass_filter(self, data, cutoff=2.5, fs=SAMPLE_FREQ, order=2):
        """
        Aplica un filtro Butterworth de paso bajo a los datos IR.
        Útil para suavizar la señal y eliminar ruido de alta frecuencia.
        """
        nyq = 0.5 * fs  # Frecuencia de Nyquist
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def butter_bandpass_filter(self, data, lowcut=0.7, highcut=3.0, fs=SAMPLE_FREQ, order=4):
        """
        Aplica un filtro Butterworth de paso banda a los datos IR.
        Esencial para aislar la frecuencia del pulso (0.7 Hz a 3.0 Hz).
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    
    def classify_bpm(self, bpm):
        """
        Clasifica el estado de la frecuencia cardíaca basado en el valor de BPM.
        """
        if bpm == 0:
            return "No hay lectura"
        elif bpm < 60:
            return "Bradicardia"
        elif bpm > 90:
            return "Taquicardia"
        else:
            return "Normal"

    def export_csv(self, auto=False):
        """
        Exporta los datos de medición recolectados a un archivo CSV.
        Si auto=True, guarda en una carpeta predeterminada con nombre de archivo con marca de tiempo.
        De lo contrario, abre un diálogo de guardado.
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
            print(f"Datos exportados a: {file_name}")
        except Exception as e:
            print(f"Error al exportar CSV: {e}")

    def calc_bpm(self, ir_data_segment):
        """
        Calcula el BPM a partir de un segmento de datos IR usando detección de picos.
        Aplica filtros para una mejor detección de picos.
        """
        if len(ir_data_segment) < SAMPLE_FREQ * 2: # Necesitamos al menos 2 segundos de datos para el cálculo
            return 0.0

        # Aplica filtro de paso banda. Este es crucial para aislar la señal del pulso.
        filtered_data = self.butter_bandpass_filter(
            ir_data_segment, lowcut=0.7, highcut=3.0, fs=SAMPLE_FREQ, order=4
        )
        
        # Determinar si necesitas invertir la señal.
        # Si la gráfica de IR crudo BAJA durante un pulso, descomenta la primera línea.
        # Si la gráfica de IR crudo SUBE durante un pulso, usa la segunda línea.
        # Observa el comportamiento de tu gráfica para decidir.
        # data_to_process = -filtered_data # Invertir para detectar valles como picos
        data_to_process = filtered_data   # No invertir, detectar picos directos

        # Asegúrate de que la señal tiene suficiente varianza para detectar picos
        if np.std(data_to_process) < 50: # Umbral de desviación estándar para ruido
            # print("Advertencia: Señal demasiado plana, posible ruido.")
            return 0.0

        # --- Detección de picos con scipy.signal.find_peaks ---
        # distance: Distancia mínima entre picos (en número de muestras).
        #           Ajusta según tu SAMPLE_FREQ y el rango de BPM esperado.
        #           Un BPM de 200 (máximo esperado) son 3 latidos por segundo.
        #           Para 100Hz, 100/3 = ~33 muestras. Usamos 0.35s para ser seguros (35 muestras).
        # prominence: Importancia del pico respecto a sus vecinos. Crucial para eliminar ruido.
        #             Ajusta basándose en la amplitud de tu señal filtrada.
        
        # Los parámetros de `prominence` pueden variar mucho. `np.std(data_to_process) * X`
        # es un buen punto de partida. Si tienes muchos picos falsos, aumenta X.
        # Si no detecta picos reales, disminuye X.
        peaks, properties = find_peaks(
            data_to_process, 
            distance=int(SAMPLE_FREQ * 0.35), # Mínimo 0.35 segundos entre picos (aprox. 171 BPM máx)
            prominence=np.std(data_to_process) * 0.4 # Ajustar este factor (0.4 es un buen valor inicial)
        )

        # print(f"Picos encontrados: {len(peaks)}, Índices de picos: {peaks}")
        # print(f"Prominencias: {properties.get('prominences', 'N/A')}")

        if len(peaks) < 2:
            return 0.0 # Necesitamos al menos dos picos para calcular un intervalo

        # Calcular los intervalos entre picos y luego el BPM promedio
        peak_intervals_samples = np.diff(peaks) # Diferencias entre los índices de los picos (en muestras)
        
        # Eliminar intervalos muy cortos o muy largos que puedan ser ruido o artefactos
        # (ej. < 40 BPM o > 180 BPM)
        min_interval_samples = SAMPLE_FREQ * (60 / 180) # Para 180 BPM
        max_interval_samples = SAMPLE_FREQ * (60 / 40) # Para 40 BPM

        valid_intervals = peak_intervals_samples[
            (peak_intervals_samples >= min_interval_samples) & 
            (peak_intervals_samples <= max_interval_samples)
        ]

        if len(valid_intervals) < 1:
            return 0.0 # No hay intervalos válidos

        avg_interval_samples = np.mean(valid_intervals)

        # BPM = (muestras por minuto) / (muestras por latido)
        # BPM = (SAMPLE_FREQ * 60) / avg_interval_samples
        bpm = (SAMPLE_FREQ * 60) / avg_interval_samples
        
        # print(f"Intervalos de picos (muestras): {peak_intervals_samples}")
        # print(f"Intervalo promedio (muestras): {avg_interval_samples}")
        # print(f"BPM calculado (bruto): {bpm}")

        # Añadir un umbral de BPM razonable para evitar valores erróneos extremos
        if 40 <= bpm <= 180: # Rango de BPM esperado para humanos
            return bpm
        else:
            return 0.0


    def closeEvent(self, event):
        """
        Maneja el evento de cierre de la aplicación.
        Apaga correctamente el sensor y el hilo de lectura.
        """
        print("Cerrando aplicación...")
        self.running = False  # Establecer bandera para detener el bucle del hilo
        if self.thread.is_alive():
            print("Esperando a que el hilo del sensor termine...")
            self.thread.join(timeout=2) # Dar al hilo una oportunidad para terminar limpiamente

        if self.sensor:
            self.sensor.shutdown() # Asegúrate de que tu clase MAX30102 tenga este método
            print("Sensor MAX30102 apagado.")
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