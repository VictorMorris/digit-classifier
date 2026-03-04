from network import Network
from readdata import MnistDataloader

training_images_filepath = 'input/train-images.idx3-ubyte'
training_labels_filepath = 'input/train-labels.idx1-ubyte'
test_images_filepath = 'input/t10k-images.idx3-ubyte'
test_labels_filepath = 'input/t10k-labels.idx1-ubyte'

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

network = Network([784, 16, 16, 10])

print(network.feed_forward(x_test[500]))
