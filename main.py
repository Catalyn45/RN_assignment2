import gzip
import numpy
import pickle
from dataclasses import dataclass


@dataclass()
class Model:
    perceptrons: numpy.array = numpy.zeros((10, 784))
    bias: numpy.array = numpy.zeros(10)


def activation(value):
    return 0 if value < 0 else 1


def train(model, train_set, learning_rate=0.09, max_iterations=-1, min_success_rate=0.75):
    images, labels = train_set
    all_classified = False
    iterations = 0

    while not all_classified:
        iterations += 1

        print(f'iteration: {iterations}')
        all_classified = True

        correctly_classified = 0
        for image_index, (image, label) in enumerate(zip(images, labels)):
            print(f'images passed: {image_index}') if image_index % 10000 == 0 else None

            t = numpy.zeros(10)
            t[label] = 1

            z = numpy.sum(model.perceptrons * numpy.array(image), axis=1) + model.bias

            output = numpy.vectorize(activation)(z)

            model.perceptrons = model.perceptrons + numpy.multiply(image.reshape((1, 784)).repeat(10, axis=0),
                                                                   ((t - output) * learning_rate)[:, numpy.newaxis])
            model.bias = model.bias + (t - output) * learning_rate

            if not numpy.array_equal(t, output):
                all_classified = False
            else:
                correctly_classified += 1

        success_rate = round(correctly_classified / 50000 * 100)
        print(
            f'correctly classified: {correctly_classified} Success rate: {success_rate}%')

        if success_rate / 100 >= min_success_rate:
            break

        if max_iterations != -1 and iterations >= max_iterations:
            break

    print("Training complete")


def test(model, test_set):
    success = 0
    failed = 0

    test_images, test_labels = test_set

    for image_index, (image, label) in enumerate(zip(test_images, test_labels)):
        z = numpy.sum(model.perceptrons * image, axis=1) + model.bias

        outputs = numpy.vectorize(activation)(z)

        numbers = numpy.where(outputs == 1)[0]

        max_chance = None

        for number in numbers:
            if max_chance is None or z[number] > z[max_chance]:
                max_chance = number

        if max_chance == label:
            success += 1
        else:
            failed += 1

    print(f'Success rate: {round(success / (success + failed) * 100)}%')


def get_datasets(file_name):
    with gzip.open(file_name, "rb") as file:
        return pickle.load(file, encoding='latin')


def main():
    train_set, valid_set, test_set = get_datasets("mnist.pkl.gz")

    model = Model()

    train(model, train_set, learning_rate=0.01, max_iterations=50, min_success_rate=0.8)
    test(model, valid_set)


if __name__ == '__main__':
    main()
