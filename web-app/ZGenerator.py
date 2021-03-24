import os
import tensorflow as tf
import numpy as np
import mcubes

from ops import *

class ZGenerator:
    def __init__(self, sess, z_dim=128, ef_dim=32, gf_dim=128, dataset_name=None):
        self.sess = sess
        
        self.input_size = 64

        self.z_dim = z_dim
        self.ef_dim = ef_dim
        self.gf_dim = gf_dim

        self.dataset_name = dataset_name

        self.real_size = 64
        self.test_size = 32
        self.batch_size = self.test_size*self.test_size*self.test_size

        self.build_model()

    def build_model(self):
        self.z_vector = tf.placeholder(shape=[1,self.z_dim], dtype=tf.float32)
        self.point_coord = tf.placeholder(shape=[self.batch_size,3], dtype=tf.float32)
        self.point_value = tf.placeholder(shape=[self.batch_size,1], dtype=tf.float32)
        
        self.zG = self.generator(self.point_coord, self.z_vector, phase_train=True, reuse=False)
        
        self.loss = tf.reduce_mean(tf.square(self.point_value - self.zG))
        
        self.saver = tf.train.Saver(max_to_keep=10)

    def generator(self, points, z, phase_train=True, reuse=False):
        with tf.variable_scope('simple_net') as scope:
            if reuse:
                scope.reuse_variables()

            zs = tf.tile(z, [self.batch_size,1])
            pointz = tf.concat([points,zs],1)

            h1 = lrelu(linear(pointz, self.gf_dim*16, 'h1_lin'))
            h1 = tf.concat([h1,pointz],1)

            h2 = lrelu(linear(h1, self.gf_dim*8, 'h4_lin'))
            h2 = tf.concat([h2,pointz],1)

            h3 = lrelu(linear(h2, self.gf_dim*4, 'h5_lin'))
            h3 = tf.concat([h3,pointz],1)

            h4 = lrelu(linear(h3, self.gf_dim*2, 'h6_lin'))
            h4 = tf.concat([h4,pointz],1)

            h5 = lrelu(linear(h4, self.gf_dim, 'h7_lin'))
            h6 = tf.nn.sigmoid(linear(h5, 1, 'h8_lin'))

            return tf.reshape(h6, [self.batch_size,1])

    def test(self, checkpoint_dir, batch_z, dim=64):
        could_load, checkpoint_counter = self.load(checkpoint_dir)
        if could_load:
            print(' [*] Load SUCCESS')
        else:
            print(' [!] Load failed...')
            return

        dima = self.test_size
        multiplier = int(dim/dima)
        multiplier2 = multiplier*multiplier
        multiplier3 = multiplier*multiplier*multiplier

        aux_x = np.zeros([dima,dima,dima],np.int32)
        aux_y = np.zeros([dima,dima,dima],np.int32)
        aux_z = np.zeros([dima,dima,dima],np.int32)
        for i in range(dima):
            for j in range(dima):
                for k in range(dima):
                    aux_x[i,j,k] = i*multiplier
                    aux_y[i,j,k] = j*multiplier
                    aux_z[i,j,k] = k*multiplier
        coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    coords[i*multiplier2+j*multiplier+k,:,:,:,0] = aux_x+i
                    coords[i*multiplier2+j*multiplier+k,:,:,:,1] = aux_y+j
                    coords[i*multiplier2+j*multiplier+k,:,:,:,2] = aux_z+k
        coords = (coords+0.5)/dim*2.0-1.0
        coords = np.reshape(coords,[multiplier3,self.batch_size,3])

        for t in range(batch_z.shape[0]):
            model_float = np.zeros([dim+2,dim+2,dim+2],np.float32)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i*multiplier2+j*multiplier+k
                        model_out = self.sess.run(self.zG,
                            feed_dict={
                                self.z_vector: batch_z[t:t+1],
                                self.point_coord: coords[minib],
                            })
                        model_float[aux_x+i+1,aux_y+j+1,aux_z+k+1] = np.reshape(model_out, [dima,dima,dima])

            thres = 0.2
            vertices, triangles = mcubes.marching_cubes(model_float, thres)

            return vertices, triangles

    def load(self, checkpoint_dir):
        import re
        print(' [*] Reading checkpoints...')

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer('(\d+)(?!.*\d)',ckpt_name)).group(0))
            print(' [*] Success to read {}'.format(ckpt_name))
            return True, counter
        else:
            print(' [*] Failed to find a checkpoint')
            return False, 0
