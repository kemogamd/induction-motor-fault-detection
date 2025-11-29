# Induction Motor Fault Detection Using Machine Learning

A complete predictive maintenance system for induction motors, combining a **Simulink digital twin**, **synthetic time-series generation**, and **real accelerometer data** collected via **ESP32**.  
The project benchmarks seven machine learning models for early-stage fault identification and achieves **92–100% accuracy** across electrical and mechanical motor faults.

---

## 🚀 Key Features

- **Digital Twin Model (Simulink):**  
  Full 4 kW induction motor model generating over **250,000 labeled samples** for multiple fault conditions.

- **Machine Learning Pipeline:**  
  Implemented and benchmarked 7 models:  
  - XGBoost  
  - MLP  
  - SVM  
  - Random Forest  
  - 1D CNN  
  - ResNet1D  
  - CNN–LSTM hybrid model  

- **Real Sensor Integration (ESP32):**  
  Real-time vibration acquisition using an IMU accelerometer.  
  Data streamed over serial → Python preprocessing → ML inference.

- **Real-Time Fault Classification:**  
  Edge-deployable fault detection with sub-second inference time.

---

## 📂 Project Structure

induction-motor-fault-detection/
│── src/
│ ├── data_preprocessing.py
│ ├── feature_extraction.py
│ ├── train_models.py
│ ├── evaluate_models.py
│ └── realtime_inference.py
│
│── simulink_model/
│ └── induction_motor_digital_twin.slx
│
│── esp32/
│ └── accelerometer_data_logger.ino
│
│── models/
│ └── saved_weights/ (optional)
│
│── data/
│ └── sample/ (small example datasets)
│
│── docs/
│ ├── system_architecture.png
│ ├── model_performance.png
│ └── readme_images/
│
│── requirements.txt
└── README.md


---

## 🧠 Machine Learning Workflow

1. **Signal generation** using Simulink digital twin.  
2. **Data preprocessing:**  
   - filtering  
   - segmentation  
   - statistical & frequency-domain feature extraction  
3. **Model training** using Scikit-Learn / PyTorch / TensorFlow.  
4. **Model evaluation** using accuracy, F1-score, and confusion matrices.  
5. **Real-time inference** using ESP32 vibration data → Python pipeline.

---

## 📊 Results

- Achieved **92–100% classification accuracy** across all fault categories.  
- ResNet1D and CNN–LSTM provided best overall performance.  
- System validated on both **synthetic** and **real ESP32 data**.

---

## 🛠️ Tech Stack

- **Python**, **NumPy**, **Pandas**, **Scikit-Learn**, **PyTorch/TensorFlow**  
- **MATLAB Simulink**  
- **ESP32**, **I2C Accelerometer (MPU6050 or similar)**  
- **Serial Communication**  
- **Git/GitHub**

---

## ▶️ How to Run

### **1. Install dependencies**
pip install -r requirements.txt


### **2. Run preprocessing**


python src/data_preprocessing.py


### **3. Train models**


python src/train_models.py


### **4. Evaluate**


python src/evaluate_models.py


### **5. Real-time inference with ESP32**
1. Upload the code under `esp32/` to your ESP32.  
2. Connect via serial:  


python src/realtime_inference.py


---

## 📌 Future Work

- Add deployment on microcontrollers (TinyML).  
- Improve synthetic-to-real domain adaptation.  
- Add GUI-based dashboard.

---

## 📄 License

Licensed under the MIT License.

---

## 👤 Author

**Kareem Hussein**  
Electrical Engineering | Predictive Maintenance & ML  
GitHub: https://github.com/kemogamd  
Email: khaledokareem@gmail.com
