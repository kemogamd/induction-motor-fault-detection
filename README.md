# Induction Motor Fault Detection Using Machine Learning

This repository contains the full implementation of my BSc Thesis project, which develops a complete predictive maintenance system for induction motors. The system integrates a Simulink digital twin, synthetic time-series generation, a custom analog vibration signal-conditioning PCB, and a Python machine learning pipeline.  
Seven ML models were benchmarked for early-stage fault identification, achieving 92–100% classification accuracy across electrical and mechanical motor faults.

---

## Key Features

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
  Real-time vibration acquisition using an ADXL335 analog accelerometer.  
  Data streamed over serial → Python preprocessing → ML inference.

- **Custom Hardware (KiCad):**  
  Designed a Vibration Signal Conditioning Board featuring:  
  - ADXL335 accelerometer  
  - AC-coupling and bias reference  
  - MCP6002 op-amp buffer  
  - Output stage optimized for ESP32 ADC input  

- **Real-Time Fault Classification:**  
  Edge-deployable fault detection with sub-second inference time.

---

## Project Structure

induction-motor-fault-detection/  
├── ml_pipeline/  
│   ├── general.py  
│   ├── PreprocessDataset.py  
│   ├── Train_CNN.py  
│   ├── Train_CNN_LSTM.py  
│   ├── Train_MLP.py  
│   ├── Train_ResNet1D.py  
│   ├── Train_Rf.py  
│   ├── Train_SVM.py  
│   └── Train_XGBoost.py  
│  
├── simulink_model/  
│   └── induction_motor_digital_twin.slx  
│  
├── hardware/  
│   └── Vibration Signal Conditioning Board KICAD/  
│       ├── Vibration Signal Conditioning Board.kicad_pro  
│       ├── Vibration Signal Conditioning Board.kicad_sch  
│       ├── Vibration Signal Conditioning Board.kicad_pcb  
│       ├── Vibration Signal Conditioning Board.kicad_prl  
│       ├── New_Library.kicad_sym  
│       └── backups/  
│  
├── esp32/  
│   └── accelerometer_data_logger.ino  
│  
├── requirements.txt  
└── README.md  

---

## Machine Learning Workflow

1. **Signal generation** using Simulink digital twin.  
2. **Data preprocessing:**  
   - filtering  
   - segmentation  
   - statistical and frequency-domain feature extraction  
3. **Model training** using Scikit-Learn / PyTorch / TensorFlow.  
4. **Model evaluation** using accuracy, F1-score, and confusion matrices.  
5. **Real-time inference** using ESP32 vibration data → Python pipeline.

---

## Results

- Achieved **92–100% classification accuracy** across all fault categories.  
- ResNet1D and CNN–LSTM provided the best overall performance.  
- System validated using both synthetic and real ESP32 vibration data.

---

## Tech Stack

- **Python**, **NumPy**, **Pandas**, **Scikit-Learn**, **PyTorch/TensorFlow**  
- **MATLAB Simulink**  
- **ESP32**, **ADXL335 accelerometer**  
- **Serial Communication**  
- **Git/GitHub**

---

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run preprocessing
python ml_pipeline/PreprocessDataset.py

### 3. Train models
python ml_pipeline/Train_ResNet1D.py  
(or any other model script)

### 4. Evaluate models
Each training script outputs metrics and logs.

### 5. Real-time inference with ESP32
1. Upload the code under `esp32/` to your ESP32.  
2. Connect via serial and run the inference script (if used).

---

## Future Work

- Deploy models on microcontrollers (TinyML).  
- Improve synthetic-to-real domain adaptation.  
- Add a GUI-based real-time monitoring dashboard.

---

## Author

**Kareem Hussein**  
Electrical Engineering | University of Debrecen  
GitHub: https://github.com/kemogamd  
Email: khaledokareem@gmail.com
