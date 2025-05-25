# MAX30102 Pulse Monitor GUI

A Python-based graphical user interface (GUI) for real-time visualization and heart rate analysis using the **MAX30102 pulse oximeter sensor**. Designed for the **Raspberry Pi**.

✅ Tested on Raspberry Pi 3B.
✅ Tested on Raspberry Pi OS Lite (headless).
✅ Tested on Python 3.11.2.

** Important** 
Using MobaXterm for remote access and display forwarding on Raspberry Pi OS Lite (no GUI). This allows you to run the GUI application over SSH from a Windows PC, even when the Pi itself has no desktop environment.

---

## 🚀 Features

* Live IR signal plotting
* Real-time BPM calculation
* Bradycardia & Tachycardia detection
* Modular and installable as a Python package
* Data export: Saving measurement data to CSV files

---

## 📷 Screenshot

![GUI](https://github.com/hbenitez-un/max30102_gui/blob/main/media/image.png)

---

## 💾 Installation

### On Raspberry Pi (with hardware)

1.  **Install dependencies:**

    ```bash
    sudo apt update
    sudo apt install python3-full python3-venv git
    ```

2.  **Clone the repository:**

    ```bash
    git clone https://github.com/hbenitez-un/max30102_gui.git
    cd max30102_gui
    ```

3.  **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 💡 Using system-installed PyQt5 in virtual environment (Recommended)

#### Prerequisite - Install PyQt5 from System Repositories:
    
    sudo apt install python3-pyqt5 qt5-default
    
If you want to use the system-installed PyQt5 instead of installing it inside the venv (Recommended):

1.  **Exit the virtual environment:**

    ```bash
    deactivate
    ```

2.  **Recreate it with system packages:**

    ```bash
    python3 -m venv venv --system-site-packages
    source venv/bin/activate
    ```

3.  **Edit `requirements.txt` and comment out the `PyQt5` line** before installing:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Test PyQt5 installation:**

    ```bash
    python3 -c 'from PyQt5 import QtWidgets; print("PyQt5 is working!")'
    ```

5.  **Install the Package:**

There are two ways to install the package:

1.  **Standard Installation:**

    ```bash
    pip install .
    ```

2.  **For Editable Mode (Recommended for Development):**

    ```bash
    pip install --editable .
    ```
    This will install your package locally with the CLI command `max30102-gui` available, and any changes you make to the source code will be reflected immediately without re-installation.


6.  **Run the GUI:**

    ```bash
    max30102-gui
    ```

The command executes main.py automatically.

---

## 💓 Sensor Information

* **Sensor Used:** MAX30102 Pulse Oximeter (via I2C)
* **Driver:** [doug-burrell/max30102](https://github.com/doug-burrell/max30102)

### Heart Rate Classification

| Range (BPM) | Status      |
| :---------- | :---------- |
| < 60        | Bradycardia |
| 60 - 100    | Normal      |
| > 100       | Tachycardia |

---

## 🗂️ Project Structure

max30102_gui/
├── max30102_gui/
│   ├── main.py
│   └── sensor/
│       └── max30102.py
├── setup.py
├── README.md
└── requirements.txt


---

## 📄 License

[MIT License](LICENSE)
Original MAX30102 driver by Doug Burrell

---

## 📬 Contact

Created by [hbenitez](https://github.com/hbenitez-un)
If you use or adapt this project, feel free to reach out!