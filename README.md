# MAX30102 Pulse Monitor GUI

A Python-based graphical user interface (GUI) for real-time visualization and heart rate analysis using the MAX30102 pulse oximeter sensor.  
Designed for the Raspberry Pi.

Tested on Raspberry Pi 3B.

---

## 🚀 Features

- Live IR signal plotting
- Real-time BPM calculation
- Bradycardia & Tachycardia detection
- Clean and responsive PyQt5 GUI
- Modular and installable as a Python package

---

## 📷 Screenshot

![screenshot-placeholder](https://via.placeholder.com/600x300?text=MAX30102+GUI)

---

## 📦 Installation

### ✅ On Raspberry Pi (with hardware)

1. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3-full python3-venv git

2. **Clone the repository**:
   ```bash
    git clone https://github.com/hbenitez-un/max30102_gui.git
    cd max30102_gui

3. **Create a virtual environment**:
   ```bash
    python3 -m venv venv
    source venv/bin/activate

4. **Install the package**:
   ```bash
    pip install .

5. **Run the GUI**:
   ```bash
    max30102-gui


**Sensor Used**
MAX30102 Pulse Oximeter (via I2C)

Driver: doug-burrell/max30102

**Heart Rate Classification**
| Range (BPM) | Status      |
| ----------- | ----------- |
| < 60        | Bradycardia |
| 60 - 100    | Normal      |
| > 100       | Tachycardia |


**Project Structure** --->

max30102_gui/
├── max30102_gui/
│   ├── main.py
│   └── sensor/
│       └── max30102.py
├── setup.py
├── README.md
└── requirements.txt

**License**
MIT License.
Original MAX30102 driver by Doug Burrell.

📬 Contact
Created by hbenitez.
If you use or adapt this project, feel free to reach out!








