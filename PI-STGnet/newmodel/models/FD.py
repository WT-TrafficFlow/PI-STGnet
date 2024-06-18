import numpy as np

from models.inits import *


def FD(pre, mean, std, init=1):
    vf_ = 60   #自由流速度，Greenberg则为最优速度，vm，而不是vf，约为一半，30      vm = 30
    rho_ = 86.16   #阻塞密度
    vf_std = std[0] / (vf_ - mean[0])                #自由流速度的归一化
    rho_std = (rho_ - mean[1]/mean[0])/(std[1]/mean[0]/mean[0])     #阻塞密度的归一化

    VF0 = np.ones(pre.shape[-2]) * vf_std  # vf 的倒数
    RHO0 = np.ones(pre.shape[-2]) * rho_std 
    if init:
        vf = tf.get_variable(name="vf", shape=[pre.shape[-2]],
                         initializer=tf.constant_initializer(VF0), trainable=True)    #获取更新自由流速度vf这个变量，维度，初始化方式，
        rho = tf.get_variable(name="rho", shape=[pre.shape[-2]],
                          initializer=tf.constant_initializer(RHO0), trainable=True)     #获取更新组赛密度rho这个变量，维度，初始化方式，
    else:
        vf = tf.get_variable(name="vf", shape=[pre.shape[-2]],
                             initializer=tf.initializers.glorot_uniform(), trainable=True)
        rho = tf.get_variable(name="rho", shape=[pre.shape[-2]],
                              initializer=tf.initializers.glorot_uniform(), trainable=True)

    q1 = tf.multiply(pre[:, :, :, 0], rho)    #rho *v
    q2 = tf.multiply(pre[:, :, :, 0], vf)      #rho *vf的倒数
    q_fd = tf.expand_dims(tf.multiply(q1, 1 - q2), axis=-1)
    q = tf.concat([pre[:, :, :, -1:], q_fd], axis=-1)
    q = tf.layers.dense(q, units=32, activation=tf.nn.relu,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    q = tf.layers.dense(q, units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    return q
