---
title: KNN DEMO
emoji: 🔥
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
pinned: false
---

# K-Nearest Neighbors (KNN) Interactive Demo

An interactive web application demonstrating the K-Nearest Neighbors algorithm with real-time visualization and educational features.

## Features

- **Multiple Datasets**: 4 built-in datasets (Iris, Wine, Breast Cancer, Diabetes)
- **Interactive Interface**: Real-time parameter adjustment and prediction
- **Smart Visualization**: Automatic t-SNE projection for high-dimensional data
- **Dual KNN System**: Shows both algorithm-accurate and visually-consistent neighbors
- **🎛Flexible Parameters**: Adjustable K value, distance metrics, and weighting methods
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Local Installation
```bash
git clone <repository-url>
cd AIO2025M03_DEMO_KNN
pip install -r requirements.txt
python app.py
```

### Usage
1. **Select Dataset**: Choose from pre-loaded datasets or upload your own CSV/Excel file
2. **Configure Target**: Select target column and problem type (classification/regression)
3. **Set Parameters**: Adjust K value, distance metric, and weighting method
4. **Input New Point**: Enter feature values for prediction
5. **Run Prediction**: Get results with interactive visualization

## Technical Highlights

- **Algorithm Accuracy**: Uses original high-dimensional space for precise KNN calculations
- **Visual Consistency**: Applies KNN in 2D t-SNE space for intuitive visualization
- **Auto-Detection**: Automatically determines classification vs regression problems
- **Error Handling**: Robust validation and user-friendly error messages

## Requirements

- Python 3.8+
- Gradio 5.38+
- Scikit-learn
- Pandas
- NumPy
- Plotly

