# NumPy-Only Deep Neural Network from Scratch

This project implements a fully functional **deep neural network** from scratch using **NumPy only**, no external machine learning libraries like TensorFlow or PyTorch.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Background](#background)
- [Features](#features)
- [Example Usage](#example-usage)
- [Future Work](#future-work)


---

## 🧠 Overview

This project was built as a learning tool to:
- Understand the fundamentals of a neural network
- Gain hands-on experience with vectorized linear algebra using NumPy
- Avoid black-box abstraction by implementing everything from scratch

It supports multiple hidden layers, various activation functions, and both classification and regression tasks.

---

## 🖼️ Background

The reason I undertook this project was that I struggled to understand how neural networks worked. Through Kaggle courses, I was able to build several deep neural networks for a variety of tasks, but I failed to grasp the reasoning behind my code - I just knew how to fill in the blanks. Just like how it is easier to know how to apply math formulas once you know the proof behind them, I sought to understand the raw math behind a neural network. Through Andrew Ng's Deep Neural Network course on Coursera, I gained the fundamentals and built this end-to-end solution to solidify my understanding. 

## ✨ Features

- Arbitrary number of layers (depth and width)
- Fully vectorized implementation (no for-loops for propagation)
- ReLU and Sigmoid activation functions
- Cross-Entropy loss function
- Supports binary classification tasks
- Modular and easy to extend

---

## 🧪 Example Usage

In ``dnn_cat_predictor.ipynb`` I created a cat identifier using the neural network and 209 images labeled as either cat or non-cat pictures. After flattening the images into 2d vectors, I trained a 4-layer model to predict whether an image contained a cat or not, achieving 80% accuracy.

---

## ⏳ Future Work
The capabilities of the current neural network architecture are pretty limited. Here are some features I would like to add in the future:
- Dropout to prevent overfitting
- Other activation functions (e.g., Softmax for multi-classification problems)
- Mini-Batch Gradient Descent
- Other optimizers like Adam optimization


