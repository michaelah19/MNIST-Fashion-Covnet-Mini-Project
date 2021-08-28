import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = "green" if predicted_label == true_label else 'red'


    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
               
# Loading Data
with open("variable.pkl", 'rb')  as f:
    predictions, x_test, y_test = pickle.load(f)

# Plotting image, with prediction vs outcome
plt.figure(figsize=(12,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plot_image(i,predictions[i],y_test,x_test)

# plt.tight_layout()
plt.suptitle("Model prediction vs labels for subset of testset")
plt.savefig("Results.png")
