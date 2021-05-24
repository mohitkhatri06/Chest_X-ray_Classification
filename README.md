## Note: This project is done for learning no medical use.
In this project, I am classifying the X-ray images using the Ensemble learning method with three classes(Normal, Covid-19, and Viral Pneumonia). CNN models—DenseNet201, Resnet50V2, and Inceptionv3 are trained individually for making predictions and then the result is combined for applying ensemble learning. Confusion matrices are used to evaluate model performance in the proposed work. The random forest achieved 91.33% accuracy whereas the weighted average ensemble method achieved 96.6% on classification.

### Tools used: 
I have used Google Colab GPU for training and testing the models, Python 3.7 and TensorFlow 2.4.1. For the implementation of the architecture , the deep learning library of TensorFlow 2.4.1 is used.

### Dataset Description:
The dataset also had a detailed description on the Kaggle (https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) and it contains the images from the all over the world it had three classes of different size. Normal: 500 images, size: 1024 x 1024 , covid-19: 500 images, size: 256 x 256 ,viral pneumonia: 500, size: 1024 x 1024. All the images are in Portable Network Graphics (PNG) file format and L mode image, that means it is a single channel image - normally interpreted as grayscale. The number of images in each classes is balanced to prevent from the bias, in all the classes we had used 500 images and resized them to 224 x 224. Problem with multiple classes and an unbalanced dataset is more difficult than a binary classification problem. Many traditional machine learning algorithms are rendered ineffective due to the skewed distribution, especially in predicting minority class examples.

For making this model more reliable the hypermeters are chosen in the pretrained models are given as below:
 * **Pre-processing** : image resizing and normalization
 * **epoch:** 30
 * **Classes:** 3 class[Normal, Covid, Viral Pneumonia]
 * **Batch size:** 16
 * **Train and test:** 80% for train and 20% test
 * **Optimizer:** Adam
 * **Learning rate:** 0.0001
 * **Classifier activation:** softmax
 * **Loss function:** SparseCategoricalCrossentropy
 * **Performance metrics:** accuracy, sensitivity, F1-score and confusion matrix
