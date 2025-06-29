import numpy as np
from sklearn.naive_bayes import BernoulliNB
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Load dataset (only use training data)
(train_X, train_y), (_, _) = mnist.load_data()

# Flatten 28x28 images to 784-dimensional vectors
train_X = train_X.reshape(train_X.shape[0], -1)

# Normalize pixel values to range [0, 1]
train_X = train_X / 255.0

# Binarize the data (BernoulliNB expects binary features)
train_X_bin = (train_X > 0.5).astype(int)

# Train Bernoulli Naive Bayes
clf = BernoulliNB()
model = clf.fit(train_X_bin, train_y)

# -------------------------------
# Generate a digit using the trained model
# -------------------------------

# Choose a digit to generate (e.g. 8)
digit_class = 8

# Get the learned probability of each pixel being black (i.e. 1) for the chosen digit
pixel_probs = np.exp(model.feature_log_prob_[digit_class])

# Sample each pixel according to its probability
generated_image = np.random.binomial(n=1, p=pixel_probs)

# -------------------------------
# Display an actual vs generated digit
# -------------------------------

plt.figure(figsize=(10, 5))

# Actual sample of the same digit class
actual_index = np.where(train_y == digit_class)[0][0]
plt.subplot(1, 2, 1)
plt.imshow(train_X_bin[actual_index].reshape(28, 28), cmap='gray')
plt.title(f'Actual Digit: {digit_class}', fontsize=14)
plt.axis('off')

# Generated image
plt.subplot(1, 2, 2)
plt.imshow(generated_image.reshape(28, 28), cmap='gray')
plt.title(f'Generated Digit: {digit_class}', fontsize=14)
plt.axis('off')

plt.show()
