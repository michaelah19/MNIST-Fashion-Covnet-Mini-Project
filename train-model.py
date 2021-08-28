# %%
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the MNIST data and preprocessing it from API
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape,y_test.shape)

# Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# %%
# Splitting training set into training and validation and also feature scaling the images (including the test set)
# Will be using 5% of training set as validation 
x_train, x_valid, x_test= x_train[3000:]/255.0, x_train[:3000]/255, x_test/255.0
y_train, y_valid = y_train[3000:], y_train[:3000]

# Model
model = keras.models.Sequential([
    # Usually i'd reccomend doing a few convolution and pooling for larger images. 
    # however those are already reduced and simplified so just flatten and scale.

    keras.layers.Flatten(input_shape = [28,28]),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(32, activation = "relu"),

    # Since the classes are mutually exclusive (can only have 1 for each picture, use softmax)
    keras.layers.Dense(len(class_names), "softmax")
])

# Compile the model (As well as assigning hyperparameters such as cost function, which optimizer, learnign rate..etc)
model.compile(loss = "sparse_categorical_crossentropy",                 # Generic Cost functions for CNN
              optimizer = keras.optimizers.SGD(learning_rate=0.02),     # Using Stochastic Gradient descent with a 0.2 learning rate
              metrics = ['accuracy'])

# Creating model
trained_model = model.fit(x=x_train, 
                          y=y_train, 
                          epochs = 30, 
                          validation_data=(x_valid, y_valid))



# %% Saving Figures and Exporting model

# Layers Overview
keras.utils.plot_model(model, to_file="CNN model.png", show_shapes=True)
# Cost and Accuracy at each epoch
pd.DataFrame(trained_model.history).plot(figsize=(8,5))
plt.gca().set_ylim(0,1)
plt.title("Cost/Accuracy Chart for training and validation set as a function of epochs")
plt.savefig("processing.png")
 
# Exporting model and predictions
predictions = model.predict(x_test)
model.save('Keras_CNN_model.h5')

with open("variable.pkl", "wb")  as f:
    pickle.dump([predictions, x_test, y_test],f)
