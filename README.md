# Waste Classification CNN System

A CNN-based waste classification system leveraging machine learning and TensorFlow to automatically identify and categorize waste type using computer vision, deployed through a Flask-powered web interface.

---

## Overview

This project implements a deep learning-based solution for automated waste classification using image data. It uses a Convolutional Neural Network (CNN) trained on labeled waste images to identify and categorize different types of waste. The goal is to demonstrate how machine learning can be applied to real-world problems such as waste segregation and environmental sustainability.

The system includes a Flask-based web interface that allows users to upload images and receive predictions in real time, making the model easy to interact with. The project is structured in a modular way, separating training, prediction, and application logic, which makes it easier to maintain, extend, and improve in the future.

---

## Features

- Image-based waste classification using a CNN model  
- Web interface built with Flask  
- Prediction from uploaded images  
- Trained model stored and loaded using TensorFlow/Keras  
- Modular code structure for training and inference  

---

## Technologies

- Python  
- TensorFlow / Keras  
- OpenCV  
- Flask
- HTML
- CSS

---

## Implementation of 7 Classes 

The model is designed to classify waste into seven distinct categories: Glass, Metal, Organic, Paper, Plastic, Textile, and E-Waste. Each class represents a common type of waste encountered in real-world segregation scenarios. Glass includes bottles and transparent materials, while Metal covers items such as cans and metallic objects. Organic consists of biodegradable waste like food scraps, whereas Paper includes recyclable paper-based materials. Plastic represents various synthetic packaging materials, and Textile includes fabric-related waste such as clothes. E-Waste covers electronic items and components that require specialized disposal. These classes enable the system to perform structured and practical waste categorization aligned with real-world recycling processes.

---

## Accuracy Fluctuation 

The model is designed to classify waste into seven categories to maintain a balance between classification accuracy and practical usability. Limiting the number of classes helps reduce model complexity and improves generalization, especially when working with a moderate-sized dataset. These categories were selected to represent commonly encountered waste types in real-world segregation systems. This approach ensures the model remains efficient while still covering a meaningful range of recyclable and non-recyclable materials.

### Why Limit the Model to 7 Classes?
  * Reduces model complexity, making training and inference more efficient.
  * Improves accuracy by avoiding over-fragmentation of classes.
  * Aligns with commonly used real-world waste segregation categories.
  * Works well with limited or moderately sized datasets.
  * Minimizes class imbalance issues during training

---

## Project Structure

Waste-Classification-CNN-System/
│
├── app.py
├── predict.py
├── train.py
├── requirements.txt
├── README.md
│
├── model/
│ └── waste_classifier.h5
│
├── static/
├── templates/
│
└── demo/

---
