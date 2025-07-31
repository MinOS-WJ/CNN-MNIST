Project Overview: Handwritten Digit Recognition System

This project is an end-to-end handwritten digit recognition system built with PyTorch for model training and Tkinter for GUI deployment. It leverages a convolutional neural network (CNN) trained on the MNIST dataset to recognize digits (0-9) from user-uploaded images. Below is a detailed breakdown:

1. Core Components

• Deep Learning Model:  

  • Architecture: 4-layer CNN (convolutional layers with 32, 64, 128, and 256 filters) → Max pooling → Dropout → Fully connected layers.  

  • Training:  

    ◦ Dataset: MNIST (28×28 grayscale digits).  

    ◦ Hyperparameters:  

      ◦ Batch size: 256  

      ◦ Epochs: 32  

      ◦ Optimizer: Adam (LR=0.001)  

      ◦ Loss: Cross-entropy  

      ◦ Regularization: Dropout (50%) and LR scheduling via ReduceLROnPlateau.  

    ◦ Metrics: Loss/accuracy tracking for train/test sets per epoch.  

  • Output: Saved best/final models (.pth files) and training logs (CSV).  

• GUI Application:  

  • Framework: Tkinter (Python's standard GUI toolkit).  

  • Functionality:  

    ◦ Model loading (supports user-selected or default paths).  

    ◦ Image upload (PNG/JPG/JPEG).  

    ◦ Real-time digit prediction with visual preview.  

  • User Flow:  

    Load Model → Upload Image → Display Prediction + Thumbnail.  

2. Key Features

• Training Pipeline:  

  • Automates data download, augmentation (normalization), training, and evaluation.  

  • Saves models and metrics in timestamped directories for reproducibility.  

  • Visualizes training curves (loss/accuracy/time) as high-res PNGs.  

• Deployment-Ready GUI:  

  • Responsive design with status indicators.  

  • Handles model/image loading errors gracefully.  

  • Default model path pre-check (results/run_*/best_model.pth).  

3. Performance Highlights

• Accuracy (from Excel logs):  

  • Peak test accuracy: 99.48% (epoch 31).  

  • Final test accuracy: 99.47% (epoch 32).  

• Efficiency:  

  • Average epoch time: 30–36 seconds (GPU-accelerated).  

4. Technical Stack

• Libraries:  

  • PyTorch, TorchVision (model training).  

  • PIL/Pillow (image preprocessing).  

  • Pandas (metric logging), Matplotlib (visualization).  

• Hardware: Auto-detects CUDA GPU support for accelerated training.  

5. Usage Workflow

1. Train the model:  
   python train_script.py  # Saves models/results in `results/run_<timestamp>`
   
2. Launch the GUI:  
   python app.py  # Loads model → Upload image → Predict digit
   
    Example GUI Interface

Summary

This project demonstrates a production-ready digit recognition system combining modern deep learning (CNN training on MNIST) and user-friendly deployment (Tkinter GUI). Its modular design separates training and inference, ensuring flexibility for model updates and edge deployment. The CNN achieves >99.4% test accuracy, validated by extensive metrics logging and visualization.
