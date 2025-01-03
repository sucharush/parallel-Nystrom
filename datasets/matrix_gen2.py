import pandas as pd
import numpy as np
import bz2
from sklearn.metrics.pairwise import rbf_kernel

def parse_sparse_mnist(row):
    # Initialize a 28x28 grid of zeros
    image = np.zeros(784, dtype=int)  # 784 = 28x28
    # Split the string on spaces to separate each index:value pair
    pixel_data = row.split()
    # Iterate over each pair
    for pixel in pixel_data:
        if ':' in pixel:  # to avoid any potential issues with malformed data
            index, value = map(int, pixel.split(':'))
            image[index] = value
    return image/255

# Load the dataset
with bz2.open("data/mnist.bz2", "rt") as f:
    df = pd.read_csv(f, header=None, nrows=40000)
df_images = df[0].apply(parse_sparse_mnist)
df_images = pd.DataFrame(df_images.tolist(), index=df_images.index)
np.save('mnist.npy', df_images.values)

def parse_libsvm_line(line):
    elements = line.strip().split()
    label = elements[0]
    indices = []
    values = []
    for elem in elements[1:]:
        index, value = elem.split(":")
        indices.append(int(index) - 1)  # LIBSVM indices are 1-based, convert to 0-based
        values.append(float(value))
    return label, indices, values

def read_libsvm_bz2(filepath, rows):
    data = []
    labels = []
    with bz2.open(filepath, 'rt') as file:
        for i, line in enumerate(file):
            if i >= rows:
                break
            label, indices, values = parse_libsvm_line(line)
            labels.append(label)
            feature_array = np.zeros(90)
            feature_array[indices] = values
            data.append(feature_array)
    return pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(90)]), labels
features_df, _ = read_libsvm_bz2("data/YearPredictionMSD.bz2", 40000)
np.save('year_prediction.npy', features_df.values)