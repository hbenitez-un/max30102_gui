# MAX30102 Pulse Monitor GUI

A Python-based graphical user interface (GUI) for real-time visualization and heart rate analysis using the **MAX30102 pulse oximeter sensor**. Designed for the **Raspberry Pi**.

✅ Tested on Raspberry Pi 3B

---

## 🚀 Features

* Live IR signal plotting
* Real-time BPM calculation
* Bradycardia & Tachycardia detection
* Clean and responsive PyQt5 GUI
* Modular and installable as a Python package

---

## 📷 Screenshot

*(You'll want to insert your screenshot here. GitHub will automatically display an image if you put the direct link or path to it.)*

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
    git clone [https://github.com/hbenitez-un/max30102_gui.git](https://github.com/hbenitez-un/max30102_gui.git)
    cd max30102_gui
    ```

3.  **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 💡 Using system-installed PyQt5 in virtual environment

If you want to use the system-installed PyQt5 instead of installing it inside the venv:

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

5.  **Install the package:**

    ```bash
    pip install .
    ```

6.  **Run the GUI:**

    ```bash
    max30102-gui
    ```

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