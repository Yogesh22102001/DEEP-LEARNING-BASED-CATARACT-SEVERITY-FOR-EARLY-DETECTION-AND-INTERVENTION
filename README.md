Retinal Cataract Detection using Deep Learning
This project aims to develop a deep learning model for detecting cataracts in retinal images. The model is built using the VGG19 architecture and fine-tuned to classify images as either "Cataract" or "Normal." The dataset is sourced from the ODIR-5K dataset, and various image preprocessing and augmentation techniques are applied to improve model performance.

Project Overview
The model uses a convolutional neural network (CNN) based on the pre-trained VGG19 architecture to classify retinal fundus images. It first preprocesses and labels the images (based on their diagnosis), then trains and evaluates the model using the labeled dataset. The model's performance is evaluated using accuracy, confusion matrix, and classification reports.

Key steps in the project:

Data Preparation: Images and labels are extracted from the dataset. Images are preprocessed (resized and normalized) for training.
Model Architecture: VGG19 pre-trained on ImageNet is used as a feature extractor, and a dense layer is added for binary classification.
Training: The model is trained using the Adam optimizer and binary cross-entropy loss.
Evaluation: Model performance is evaluated using metrics such as accuracy, confusion matrix, and classification report.
Visualization: Plots showing training/validation accuracy and loss over epochs, as well as image predictions on the test set.
Features
Data Preprocessing:

Images are resized to a consistent size of 224x224 pixels.
Labels are extracted based on the diagnostic keywords indicating cataract presence.
Model Architecture:

VGG19 is used as a feature extractor with frozen layers.
A fully connected layer with a sigmoid activation function for binary classification (Cataract vs. Normal).
Training & Evaluation:

The model is trained using the Adam optimizer with binary cross-entropy loss.
Model performance is evaluated using accuracy, confusion matrix, and classification report.
Early stopping and model checkpoints are implemented for better training and preventing overfitting.
Visualization:

Displays sample images along with their predicted and actual labels.
Visualizes training/validation accuracy and loss curves.
Requirements
Python 3.x
TensorFlow 2.x
Keras
OpenCV
NumPy
Pandas
Matplotlib
scikit-learn
mlxtend
Installation
Clone the repository and install the required dependencies using pip:

bash
Copy
git clone https://github.com/Yogesh22102001/cataract-detection.git
cd cataract-detection
pip install -r requirements.txt
Dataset
The dataset used in this project is the ODIR-5K dataset, which contains labeled retinal images used for ocular disease detection. The preprocessed images can be accessed here.

Left and Right Fundus Images: Used for the detection of cataracts.
Labels: The dataset contains diagnostic keywords, indicating whether cataracts are present in the images.
Usage
Preprocessing: The dataset is preprocessed and labeled into two categories—Cataract and Normal.

Model Training:

python
Copy
# Train the model
history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), callbacks=[checkpoint, earlystop])
Model Evaluation:

python
Copy
# Evaluate the trained model
loss, accuracy = model.evaluate(x_test, y_test)
Visualizing Results:

Accuracy and loss curves are plotted.
Sample predictions are displayed alongside ground truth labels.
Results
Model Accuracy: The model achieves high accuracy, with further evaluation performed using confusion matrix and classification report.
Confusion Matrix: The confusion matrix is used to visualize the performance across both classes: Normal and Cataract.
