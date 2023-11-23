### Auto-Encoder:
In this project, we will focusing on theory of Auto-encoders and its implementation. 
### The auto-encoders are:
- Unsupervised Learning
- Compress data
- Used for Compression, Denosing and Detection

### Layer Architecture of Auto-Encoder
![download](https://github.com/aliabbas30/Auto_Encoders_Example/assets/102746791/ebdc8def-917d-4926-85a8-e84fe43155bc)

### Latent Space:
The latent space refers to a lower-dimensional space in which data points are represented after being processed by an encoder in a machine learning model, such as an autoencoder. In the context of autoencoders, the latent space is the internal, compact representation of input data that the model has learned during the training process.

### Loss in Auto encoders
**Mean squared error (MSE)** is a common loss function used in autoencoders because it is a straightforward and effective way to measure the difference between the input and output of the autoencoder. The goal of an autoencoder is to learn a compressed representation of the input data, and MSE helps to ensure that the compressed representation is as close to the original input as possible.

MSE is calculated by taking the average of the squared differences between the input and output values. The lower the MSE, the closer the output is to the input. MSE is a good choice for autoencoders because it is easy to compute and is sensitive to small changes in the input and output values.

However, there are other loss functions that can be used in autoencoders, depending on the specific application. Some alternative loss functions include:

**Binary cross-entropy (BCE):** BCE is a good choice for binary data, such as images or text. It measures the difference between the probability that a given pixel or word is zero and the actual value of the pixel or word.


**Kullback-Leibler (KL) divergence:** KL divergence is a measure of the difference between two probability distributions. It can be used to ensure that the compressed representation of the input data is as similar as possible to the original distribution of the data.


**Structural similarity index (SSIM):** SSIM is a good choice for images, as it measures the similarity between two images in terms of their structural information.


The choice of loss function depends on the specific application and the type of data being used. In general, MSE is a good starting point, but it is always worth experimenting with other loss functions to see if they improve the performance of the autoencoder.

### About Data:

**Fashion MNIST** is a widely-used benchmark dataset for testing image classification algorithms in machine learning. It comprises 28x28 pixel grayscale images of 10 different fashion items, such as T-shirts, trousers, and sneakers. With 60,000 training and 10,000 test images, Fashion MNIST serves as a more challenging alternative to the traditional handwritten digit MNIST dataset. It has become a standard in the field, aiding in the evaluation and comparison of various machine learning models for image classification tasks. The dataset is accessible through popular machine learning libraries like TensorFlow and PyTorch. If used in research, it is recommended to cite the original paper ([arXiv:1708.07747](https://arxiv.org/abs/1708.07747)).


### Used Libraries


1. **Matplotlib:**
   - Documentation: [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
   - GitHub Repository: [Matplotlib GitHub](https://github.com/matplotlib/matplotlib)

2. **NumPy:**
   - Documentation: [NumPy Documentation](https://numpy.org/doc/)
   - GitHub Repository: [NumPy GitHub](https://github.com/numpy/numpy)

3. **Pandas:**
   - Documentation: [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
   - GitHub Repository: [Pandas GitHub](https://github.com/pandas-dev/pandas)

4. **TensorFlow:**
   - Documentation: [TensorFlow Documentation](https://www.tensorflow.org/guide)
   - GitHub Repository: [TensorFlow GitHub](https://github.com/tensorflow/tensorflow)

5. **Scikit-Learn:**
   - Documentation: [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
   - GitHub Repository: [Scikit-Learn GitHub](https://github.com/scikit-learn/scikit-learn)




### To download the required libraries use requirement.txt
```bash
pip install -r requirement.txt
```

### About Auto-Encoders
Tutorial: https://www.youtube.com/watch?v=qiUEgSCyY5o&ab_channel=IBMTechnology


```python

```

