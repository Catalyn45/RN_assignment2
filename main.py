import gzip
import numpy
import pickle
from dataclasses import dataclass
import random
from scipy.special import softmax
import matplotlib.pyplot as plt


@dataclass()
class Model:
    perceptrons: numpy.array = numpy.zeros((10, 784))
    bias: numpy.array = numpy.zeros(10)


def activation(value):
    return 0 if value <= 0 else 1


def train(model, train_set, test_set, learning_rate=0.09, max_iterations=-1):
    images, labels = train_set
    samples = list(zip(images, labels))

    all_classified = False
    epoch = 0

    while not all_classified:
        all_classified = True

        random.shuffle(samples)
        for image_index, (image, label) in enumerate(samples):
            t = numpy.zeros(10)
            t[label] = 1

            z = numpy.sum(model.perceptrons * numpy.array(image), axis=1) + model.bias

            output = softmax(z) if USE_SOFTMAX else numpy.vectorize(activation)(z)

            model.perceptrons = model.perceptrons + image * ((t - output) * learning_rate)[:, numpy.newaxis]
            model.bias = model.bias + (t - output) * learning_rate

            max_indices = numpy.where(output == output.max())[0]

            max_index = max_indices[0]

            for i in range(1, len(max_indices)):
                if z[max_indices[i]] > z[max_index]:
                    max_index = i

            if max_index != label:
                all_classified = False

        rate = test(model, test_set)
        print(f'Epoch: {epoch} accuracy: {round(rate * 100, 2)} %')

        if max_iterations != -1 and epoch >= max_iterations:
            break

        learning_rate *= .98
        epoch += 1

    print("Training complete")


def test(model, test_set):
    success = 0
    failed = 0

    test_images, test_labels = test_set

    for image_index, (image, label) in enumerate(zip(test_images, test_labels)):
        z = numpy.sum(model.perceptrons * image, axis=1) + model.bias

        outputs = softmax(z) if USE_SOFTMAX else numpy.vectorize(activation)(z)

        max_indices = numpy.where(outputs == outputs.max())[0]

        max_index = max_indices[0]

        for i in range(1, len(max_indices)):
            if z[max_indices[i]] > z[max_index]:
                max_index = i

        if max_index == label:
            success += 1
        else:
            failed += 1

    rate = success / (success + failed)

    return rate


def get_datasets(file_name):
    with gzip.open(file_name, "rb") as file:
        return pickle.load(file, encoding='latin')


USE_SOFTMAX = True


def main():
    train_set, valid_set, test_set = get_datasets("mnist.pkl.gz")

    model = Model()

    train(model, train_set, valid_set, learning_rate=0.001)


if __name__ == '__main__':
    main()

