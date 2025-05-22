from PyQt5.QtWidgets import QApplication, QLabel
import sys

app = QApplication(sys.argv)
label = QLabel("¡Hello World!")
label.setWindowTitle("Hello World")
label.show()
app.exec_()
