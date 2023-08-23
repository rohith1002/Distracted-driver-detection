Project Title: Distracted Driver Detection using Various Deep Learning Models

Aim: To detect unsafe driving behaviors and potential distractions, ensuring road safety by preventing accidents and harm to others.

Dataset: State Farm Distracted Driver Detection Dataset from Kaggle

Training set: 22,424 images
Testing set: 79,726 images
Classes: 10
Image size: 640x480 pixels
Image format: JPG
Class distribution: Uneven, 2,000 to 2,800 images per class
Data source: Dashcams in vehicles driven by State Farm customers
Methods:

Data Preprocessing:
Data split based on driver IDs to prevent data leakage.
Image preprocessing using OpenCV.
Models:
PCA and Autoencoder with K-Nearest Neighbors (KNN) classifier.
Custom Convolutional Neural Network (CNN) 01.
Custom Convolutional Neural Network (CNN) 02.
VGG16 pre-trained model.
ResNet pre-trained model.
MobileNet pre-trained model.
Libraries and Frameworks Used:
Python: Programming language.
PyTorch and TensorFlow: Deep learning frameworks for model development.
Keras: High-level neural networks API (built on TensorFlow).
NumPy: Numerical computations and array operations.
Pandas: Data manipulation and analysis.
Scikit-learn: Machine learning algorithms and evaluation metrics.
OpenCV: Computer vision tasks, image preprocessing, and manipulation.
Matplotlib: Data visualization and plotting.
Jupyter Notebook: Interactive code development and experimentation.
Kaggle Notebook: Cloud-based environment for code execution and dataset exploration.
Workflow:

Data Splitting: Split data by driver IDs to prevent data leakage and ensure generalization.
Data Preprocessing: Apply image preprocessing techniques using OpenCV.
Model Development:
Implement PCA and Autoencoder for dimensionality reduction and feature extraction.
Design and train custom CNN models tailored to the problem.
Utilize pre-trained VGG16, ResNet, and MobileNet models for transfer learning.
Model Training:
Train all models on the training data while ensuring no data leakage.
Model Evaluation:
Evaluate models using appropriate metrics such as accuracy, precision, recall, etc.
Fine-Tuning and Optimization:
Perform hyperparameter tuning and optimize models for better performance.
Ensemble (Optional):
Combine the predictions from multiple models for improved results.
Testing:
Test the final models on the testing dataset to assess real-world performance.
Conclusion:
By implementing a variety of deep learning models and leveraging techniques like PCA, Autoencoder, and transfer learning from pre-trained models, the project aims to accurately detect unsafe driving behaviors and distractions. The careful data preprocessing, model selection, and training on driver-based splits mitigate data leakage issues, ensuring robust and reliable model performance. This project contributes to enhancing road safety by identifying potential accident-causing actions and enabling timely interventions.
