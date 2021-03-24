import heapq

from flask import Flask, request, render_template, jsonify
from gevent.pywsgi import WSGIServer

from MDN import *
from ZGenerator import *

app = Flask(__name__)

def mdn_predict(labels, gaussian_idx=0):
    mdn = MDN()
    predictions_alpha, predictions_mu, predictions_sigma = mdn.test(
        weights_dir = 'mdn-weights/mdn.h5',
        batch_labels = labels)

    final_mu = []
    final_sigma = []
    for i in range(predictions_alpha.shape[0]):
        idx_sorted = heapq.nlargest(5, range(len(predictions_alpha[i, 0])), key=predictions_alpha[i, 0].__getitem__)
        alpha_sorted = heapq.nlargest(5, predictions_alpha[i, 0])
        idx_selected = idx_sorted[gaussian_idx]
        alpha_selected = alpha_sorted[gaussian_idx]
        final_mu.append(predictions_mu[i, :, idx_selected])
        final_sigma.append(predictions_sigma[i, :, idx_selected])
    final_mu = np.array(final_mu)
    final_sigma = np.array(final_sigma)

    return final_mu, final_sigma

def zgenerator_predict(zeta):
    vertices = None
    triangles = None

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        zgenerator = ZGenerator(sess, dataset_name='chairs_vox')
        vertices, triangles = zgenerator.test(checkpoint_dir='zgenerator-checkpoint', batch_z=zeta)

    return vertices, triangles

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        data = request.json

        labels = np.zeros((1, 1, 25))

        if data['labels'][0] == 'back-size-half':
            labels[0, 0, 0] = 1
        if data['labels'][0] == 'back-size-full':
            labels[0, 0, 1] = 1

        if data['labels'][1] == 'back-fill-solid':
            labels[0, 0, 2] = 1
        if data['labels'][1] == 'back-fill-vertical-ladder':
            labels[0, 0, 3] = 1
        if data['labels'][1] == 'back-fill-horizontal-ladder':
            labels[0, 0, 4] = 1
        if data['labels'][1] == 'back-fill-holes':
            labels[0, 0, 5] = 1
        if data['labels'][1] == 'back-fill-no-back':
            labels[0, 0, 6] = 1

        if data['labels'][2] == 'back-side-view-straight':
            labels[0, 0, 7] = 1
        if data['labels'][2] == 'back-side-view-bent':
            labels[0, 0, 8] = 1

        if data['labels'][3] == 'back-front-view-round':
            labels[0, 0, 9] = 1
        if data['labels'][3] == 'back-front-view-curved':
            labels[0, 0, 10] = 1
        if data['labels'][3] == 'back-front-view-square':
            labels[0, 0, 11] = 1

        if data['labels'][4] == 'seat-shape-square':
            labels[0, 0, 12] = 1
        if data['labels'][4] == 'seat-shape-circular':
            labels[0, 0, 13] = 1

        if data['labels'][5] == 'leg-number-one':
            labels[0, 0, 14] = 1
        if data['labels'][5] == 'leg-number-two':
            labels[0, 0, 15] = 1
        if data['labels'][5] == 'leg-number-three':
            labels[0, 0, 16] = 1
        if data['labels'][5] == 'leg-number-four':
            labels[0, 0, 17] = 1
        if data['labels'][5] == 'leg-number-five':
            labels[0, 0, 18] = 1

        if data['labels'][6] == 'leg-length-long':
            labels[0, 0, 19] = 1
        if data['labels'][6] == 'leg-length-short':
            labels[0, 0, 20] = 1

        if data['labels'][7] == 'leg-type-roller':
            labels[0, 0, 21] = 1
        if data['labels'][7] == 'leg-type-straight':
            labels[0, 0, 22] = 1
        if data['labels'][7] == 'leg-type-beam':
            labels[0, 0, 23] = 1
        if data['labels'][7] == 'leg-type-box':
            labels[0, 0, 24] = 1

        data['samples'] = [float(i) for i in data['samples']]

        mu, sigma = mdn_predict(labels)

        sigma_indices = np.argsort(-sigma, axis=1)[0]

        zeta = mu
        for i in range(len(data['samples'])):
            zeta[0][sigma_indices[i]] = zeta[0][sigma_indices[i]] + data['samples'][i]

        vertices, triangles = zgenerator_predict(zeta)

        sigma_sorted = sigma[0][sigma_indices]
        largest = sigma_sorted[0]
        sigma_top = sigma_sorted[sigma_sorted > 0.2]

        return jsonify({'vertices': vertices.tolist(), 'triangles': triangles.tolist(), 'sigma': sigma_top.tolist()})
    return None

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
