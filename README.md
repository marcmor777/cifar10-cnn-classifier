# CIFAR-10 Image Classifier with a Convolutional Neural Network

This project demonstrates the creation, training, and evaluation of a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The primary goal is to showcase an end-to-end deep learning workflow, including data preprocessing, model building, and techniques to mitigate overfitting like **Data Augmentation** and **Dropout**.

---

## 📈 Project Results & Visuals

The model was trained using **Early Stopping** to prevent overfitting and find the optimal number of epochs. Below are the learning curves for accuracy and loss on both the training and validation datasets.

**Accuracy & Loss Plots**
*The model's validation accuracy closely follows the training accuracy, indicating that the regularization techniques (Data Augmentation and Dropout) were effective in helping the model generalize well.*

**Final Performance:**
After training, the model achieved a **final accuracy of 67.51%** on the unseen test dataset.

**Example Predictions:**
Here is an example of the model classifying a random image from the test set:

| Real Label | Predicted Label |
| :--------: | :---------------: |
|    Ship    |       Ship        |


---

## 🛠️ Technologies Used

- **Python 3**
- **TensorFlow & Keras:** For building and training the deep learning model.
- **NumPy:** For numerical operations and data manipulation.
- **Pandas:** To handle the training history for plotting.
- **Matplotlib:** For data visualization and plotting the results.

---

## 📂 Project Structure

```
.
├── notebook.ipynb         # The main Jupyter Notebook with all the code
├── README.md              # Project documentation (this file)
├── requirements.txt       # Python dependencies
└── .gitignore             # Standard Python .gitignore file
```

---

## 📖 Project Workflow

The project followed a structured machine learning workflow:

1.  **Data Loading and Preprocessing:**
    - The CIFAR-10 dataset was loaded directly from `keras.datasets`.
    - Pixel values were **normalized** from a range of [0, 255] to [0, 1] to improve training stability.
    - Labels were **one-hot encoded** to be compatible with the model's final `softmax` activation layer.

2.  **Model Architecture:**
    - A `Sequential` Keras model was built with the following key components:
        - **Data Augmentation Layers (`RandomFlip`, `RandomRotation`, `RandomZoom`):** To artificially expand the training dataset and make the model more robust to variations in the images.
        - **Two Convolutional Blocks (`Conv2D` + `MaxPooling2D`):** To act as feature extractors, identifying patterns like edges, textures, and shapes.
        - **Dropout Layers:** Added after the second convolutional block and before the output layer to randomly deactivate neurons during training, which helps prevent overfitting.
        - **Classifier Head (`Flatten` + `Dense`):** A standard fully-connected head to make the final classification decision based on the extracted features. The output layer uses a `softmax` activation for multi-class probability distribution.

3.  **Training and Evaluation:**
    - The model was compiled with the **Adam optimizer** and **categorical cross-entropy** loss function.
    - **Early Stopping** was used to monitor `val_accuracy` and stop the training process when the model ceased to improve, automatically restoring the best weights found.
    - The training history was plotted to visually inspect the model's learning process and diagnose overfitting.
    - The final model was evaluated on the test set to determine its generalization performance.

---

## 🚀 How to Run

To replicate this project, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter Notebook
jupyter notebook notebook.ipynb
```

---

## 💡 Future Improvements

While the model performs reasonably well, several enhancements could be explored to further improve accuracy:

-   **Hyperparameter Tuning:** Experiment with different numbers of filters, kernel sizes, and dropout rates to find a more optimal configuration.
-   **Transfer Learning:** Implement a more advanced approach by using a pre-trained model (like VGG16, ResNet50, or MobileNet) and fine-tuning it on the CIFAR-10 dataset. This would likely provide a significant boost in accuracy.
-   **More Complex Architecture:** Add more convolutional blocks or experiment with different architectures like residual connections to increase the model's learning capacity.