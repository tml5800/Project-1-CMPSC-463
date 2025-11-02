# 🧠 Project 1 – Time-Series Clustering and Segment Analysis on PulseDB  
**Author:** Tommy Lu  
**Course:** CMPSC 463 – Design & Analysis of Algorithms  
**Instructor:** Dr. Janghoon Yang  

---

## 🩺 Overview
This project performs **unsupervised clustering of biomedical time-series segments** from PulseDB using **divide-and-conquer algorithms**.  
Each segment (e.g., ABP/PPG/ECG) is analyzed to:
- Cluster similar signals recursively.
- Identify the **closest pair** within clusters using **Dynamic Time Warping (DTW)**.
- Detect the **most active interval** in each segment via **Kadane’s Algorithm**.  

The goal is to demonstrate how **algorithmic reasoning**, not machine learning heuristics, can reveal structure in physiological data.

---

## ⚙️ Installation & Usage

```bash
# 1. Clone this repository
git clone https://github.com/tml5800/Project-1-CMPSC-463
cd Project-1-CMPSC-463

# 2. Install required packages
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py
