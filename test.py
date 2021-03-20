import os
import argparse
import heapq
import pandas as pd

from MDN import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='chairs', help='name of the dataset (chairs/lamps/tables)')
    parser.add_argument('--labels_file', type=str, default='chairs_artificial_labels', help='file name the spreadsheet of user-provided labels saved in the data folder')
    parser.add_argument('--gaussian_idx', type=int, default=0, help='the index of Gaussian (sorted in the descending order of alpha) for generating the mu and sigma')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    labels_file = args.labels_file
    gaussian_idx = args.gaussian_idx

    if not os.path.exists(os.path.join(RESULTS_DIR, dataset_name + '_mdn.h5')):
        raise('Network has not been trained!')

    if dataset_name == 'chairs':
        mdn = MDN(labels_dim=25)
    elif dataset_name == 'lamps':
        mdn = MDN(labels_dim=24)
    elif dataset_name == 'tables':
        mdn = MDN(labels_dim=20)
    else:
        raise('Dataset is not supported!')

    labels = pd.read_excel(os.path.join(DATA_DIR, labels_file + '.xlsx'))
    labels = pd.DataFrame(labels)
    labels = labels.iloc[:, 3:]
    labels = labels.to_numpy()
    labels = np.transpose(labels)
    labels = np.expand_dims(labels, axis=1)

    alpha, mu, sigma = mdn.test(
        labels_test = labels,
        weights_dir = os.path.join(RESULTS_DIR, dataset_name + '_mdn.h5'))

    final_mu = []
    final_sigma = []
    for i in range(alpha.shape[0]):
        idx_sorted = heapq.nlargest(5, range(len(alpha[i, 0])), key=alpha[i, 0].__getitem__)
        alpha_sorted = heapq.nlargest(5, alpha[i, 0])
        idx_selected = idx_sorted[gaussian_idx]
        alpha_selected = alpha_sorted[gaussian_idx]
        final_mu.append(mu[i, :, idx_selected])
        final_sigma.append(mu[i, :, idx_selected])
    final_mu = np.array(final_mu)
    final_sigma = np.array(final_sigma)

    np.savetxt(os.path.join(RESULTS_DIR, dataset_name + '_mu.csv'), final_mu, delimiter=',')
    np.savetxt(os.path.join(RESULTS_DIR, dataset_name + '_sigma.csv'), final_sigma, delimiter=',')
