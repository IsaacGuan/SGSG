import os
import argparse
import pandas as pd

from MDN import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='chairs', help='name of the dataset (chairs/lamps/tables)')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning rate for training')
    parser.add_argument('--batch_size', type=int, default=10, help='number of training instances per batch')
    parser.add_argument('--epoch_num', type=int, default=3000, help='number of training epochs')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epoch_num = args.epoch_num

    if dataset_name == 'chairs':
        mdn = MDN(labels_dim=25)
    elif dataset_name == 'lamps':
        mdn = MDN(labels_dim=24)
    elif dataset_name == 'tables':
        mdn = MDN(labels_dim=20)
    else:
        raise('Dataset is not supported!')

    labels = pd.read_excel(os.path.join(DATA_DIR, dataset_name + '_labels.xlsx'))
    labels = pd.DataFrame(labels)
    labels = labels.iloc[:, 3:]
    labels = labels.to_numpy()
    labels = np.transpose(labels)
    labels = np.expand_dims(labels, axis=1)

    z = pd.read_csv(os.path.join(DATA_DIR, dataset_name + '_z.csv'), header=None)
    z = pd.DataFrame(z)
    if dataset_name == 'tables':
        z = z.iloc[0:100, :]
    z = z.to_numpy()
    z = np.expand_dims(z, axis=1)

    model = mdn.train(
        learning_rate = learning_rate,
        batch_size = batch_size,
        epoch_num = epoch_num,
        labels_train = labels,
        z_train = z)

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    model.save_weights(os.path.join(RESULTS_DIR, dataset_name + '_mdn.h5'))
