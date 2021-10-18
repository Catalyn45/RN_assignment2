import gzip
import numpy
import pickle
from dataclasses import dataclass
import random
from scipy.special import softmax


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
    iterations = 0

    while not all_classified:
        all_classified = True

        random.shuffle(samples)
        for image_index, (image, label) in enumerate(samples):
            t = numpy.zeros(10)
            t[label] = 1

            z = numpy.sum(model.perceptrons * numpy.array(image), axis=1) + model.bias

            output = softmax(z) if USE_SOFTMAX else numpy.vectorize(activation)(z)

            model.perceptrons = model.perceptrons + numpy.multiply(image.reshape((1, 784)).repeat(10, axis=0),
                                                                   ((t - output) * learning_rate)[:, numpy.newaxis])
            model.bias = model.bias + (t - output) * learning_rate

            if not numpy.argmax(output) == numpy.argmax(t):
                all_classified = False

        test(model, test_set, iterations)

        if max_iterations != -1 and iterations >= max_iterations:
            break

        iterations += 1

    print("Training complete")


def test(model, test_set, epoch):
    success = 0
    failed = 0

    test_images, test_labels = test_set

    for image_index, (image, label) in enumerate(zip(test_images, test_labels)):
        z = numpy.sum(model.perceptrons * image, axis=1) + model.bias

        outputs = softmax(z) if USE_SOFTMAX else numpy.vectorize(activation)(z)

        if numpy.argmax(outputs) == label:
            success += 1
        else:
            failed += 1

    print(f'Epoch: {epoch} success rate: {round(success / (success + failed) * 100)}%')


def get_datasets(file_name):
    with gzip.open(file_name, "rb") as file:
        return pickle.load(file, encoding='latin')


USE_SOFTMAX = False


def main():
    train_set, valid_set, test_set = get_datasets("mnist.pkl.gz")

    model = Model()

    train(model, train_set, valid_set, learning_rate=0.003, max_iterations=50)


if __name__ == '__main__':
    main()
