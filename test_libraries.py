print("Starting library test...\n")

# TensorFlow
import tensorflow as tf
print("TensorFlow:", tf.__version__)

# NumPy
import numpy as np
print("NumPy:", np.__version__)

# Pandas
import pandas as pd
print("Pandas:", pd.__version__)

# Matplotlib
import matplotlib
print("Matplotlib:", matplotlib.__version__)

# Scikit-learn
import sklearn
print("Scikit-learn:", sklearn.__version__)

# OpenCV
import cv2
print("OpenCV:", cv2.__version__)

# Pillow
from PIL import Image
print("Pillow: Working")

# Streamlit
import streamlit as st
print("Streamlit:", st.__version__)

print("\n--------------------------------")
print("ALL LIBRARIES WORKING CORRECTLY!")
print("--------------------------------")