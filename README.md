# GAN Studio Pro - MNIST Digit Generation with GANs

![GAN Studio Pro Demo](https://github.com/yourusername/gan-studio-pro/raw/main/demo.gif)

## Table of Contents
- [Project Description](#project-description)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Description

GAN Studio Pro is an interactive web application that demonstrates Generative Adversarial Networks (GANs) for generating realistic handwritten digits (0-9). The application features:

- A pre-trained Deep Convolutional GAN (DCGAN) model
- Real-time digit classification
- Interactive drawing canvas
- Training visualization tools

## Key Features

### 🎨 Digit Generation
- Generate grids of 1-16 synthetic digits
- Adjustable noise seed for reproducible results
- High-quality 28x28 grayscale outputs

### ✏️ Drawing Tools
- Interactive canvas with adjustable brush
- Real-time digit classification
- Multiple drawing modes (freehand, shapes)
- Customizable colors and brush sizes

### 📊 Training Visualization
- View sample outputs from training epochs
- Observe GAN convergence patterns
- Compare early vs late training results

## Tech Stack

### Backend
- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy

### Frontend
- Streamlit
- Matplotlib
- PIL/Pillow
- Streamlit-drawable-canvas

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- GPU recommended (not required)

### Quick Start
```bash
# Clone repository
git clone https://github.com/Prajwal9823/gan-studio-pro.git
cd gan-studio-pro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
