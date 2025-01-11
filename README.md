# -Cats-vs-Dogs-Image-Classification-with-Deep-Learning
A project to classify images of cats and dogs using TensorFlow and Keras. Includes data preprocessing, model training, performance evaluation (accuracy, precision, recall, F1 score), visualizations like confusion matrix and category distribution, and saving model weights and predictions.


<h2 style="font-size: 3em;">Overview</h2>



Prepares training and validation datasets using image augmentation.
Implements a Convolutional Neural Network (CNN) with three convolutional layers, batch normalization, and dropout for regularization.
Trains the model with callbacks for early stopping and learning rate reduction.
Evaluates model performance using accuracy, precision, recall, F1 score, and confusion matrix.
Visualizes sample predictions, class distributions, and learning curves.


<h2 style="font-size: 3em;">Dataset</h2>

Training Data: Images of cats and dogs stored in ../data/train/. Each image file is named with a prefix (cat or dog) to indicate its label.
Testing Data: Images for testing stored in ../data/test/.
The dataset is loaded, labels are extracted from filenames, and they are encoded as 0 (cat) and 1 (dog).


<h2 style="font-size: 3em;">Model Architecture</h2>

The Convolutional Neural Network includes:

Convolutional Layers: Three layers with ReLU activation and max pooling.
Batch Normalization: Normalizes layer inputs to stabilize training.
Dropout: Reduces overfitting.
Fully Connected Layer: Flattening followed by dense layers.
Output Layer: Softmax activation for binary classification (cat vs dog).


<h2 style="font-size: 3em;">Visualization</h2>
Training vs Validation Loss and Accuracy: Plots to monitor model performance.
Sample Augmented Images: Visual inspection of augmented training data.
Confusion Matrix: Displays the normalized confusion matrix for predictions.
Class Distribution: Bar plots for predicted categories.


<h2 style="font-size: 3em;">Result Matrix</h2>
Accuracy, Precision, Recall, and F1 Score are calculated using sklearn.
Confusion Matrix: Visualizes the relationship between true and predicted labels.
Predicted Samples: Displays test images with their predicted labels.


<h2 style="font-size: 3em;">Usuage</h2>
Place the dataset in the ../data/ directory with subfolders train/ and test/.
Run the script to train the model and generate predictions.
Evaluate the results using the provided visualizations and metrics.


<h2 style="font-size: 3em;">Refrences</h2>
TensorFlow Documentation: https://www.tensorflow.org/

Keras Documentation: https://keras.io/

Dataset: Kaggle Dogs vs Cats





