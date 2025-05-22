MAX30102 Pulse Monitor GUI
A Python-based graphical user interface (GUI) for real-time visualization and heart rate analysis using the MAX30102 pulse oximeter sensor.
Designed for the Raspberry Pi.

✅ Tested on Raspberry Pi 3B

🚀 Features
Live IR signal plotting

Real-time BPM calculation

Bradycardia & Tachycardia detection

Clean and responsive PyQt5 GUI

Modular and installable as a Python package

📷 Screenshot


💾 Installation
On Raspberry Pi (with hardware)
Install dependencies:

bash
Copiar
Editar
sudo apt update
sudo apt install python3-full python3-venv git
Clone the repository:

bash
Copiar
Editar
git clone https://github.com/hbenitez-un/max30102_gui.git
cd max30102_gui
Create a virtual environment:

bash
Copiar
Editar
python3 -m venv venv
source venv/bin/activate
💡 Using system-installed PyQt5 in virtual environment
If you want to use the system-installed PyQt5 instead of installing it inside the venv:

Exit the virtual environment:

bash
Copiar
Editar
deactivate
Recreate it with system packages:

bash
Copiar
Editar
python3 -m venv venv --system-site-packages
source venv/bin/activate
Edit requirements.txt and comment out the PyQt5 line before installing:

bash
Copiar
Editar
pip install -r requirements.txt
Test PyQt5 installation:

bash
Copiar
Editar
python3 -c "from PyQt5 import QtWidgets; print('PyQt5 is working!')"
Install the package:

bash
Copiar
Editar
pip install .
Run the GUI:

bash
Copiar
Editar
max30102-gui
💓 Sensor Information
Sensor Used: MAX30102 Pulse Oximeter (via I2C)

Driver: doug-burrell/max30102

Heart Rate Classification
Range (BPM)	Status
< 60	Bradycardia
60 - 100	Normal
> 100	Tachycardia

🗂️ Project Structure
arduino
Copiar
Editar
max30102_gui/
├── max30102_gui/
│   ├── main.py
│   └── sensor/
│       └── max30102.py
├── setup.py
├── README.md
└── requirements.txt
📄 License
MIT License
Original MAX30102 driver by Doug Burrell

📬 Contact
Created by hbenitez
If you use or adapt this project, feel free to reach out!