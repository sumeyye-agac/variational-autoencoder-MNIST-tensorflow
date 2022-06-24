import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from model import model

# initial parameters
learning_rate = 0.001
batch_size = 100
epoch_number = 50
hidden_dim = 64


# Created to plot convergence curves based on and loss values obtained during training
def plot_curves(plot_general_loss_, plot_rec_loss_, plot_kl_loss_, epoch_number):
    plot_general_loss_, plot_rec_loss_, plot_kl_loss_ = np.array(plot_general_loss_), np.array(
        plot_rec_loss_), np.array(plot_kl_loss_)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot()
    ax.set_xlabel('epochs')
    ax.tick_params()

    ax.plot(plot_general_loss_, label='general loss')
    ax.plot(plot_rec_loss_, label='reconstruction loss')
    ax.plot(plot_kl_loss_, label='kl-divergence loss')

    ax.set_ylabel("loss")
    ax.legend(loc='upper right')

    plt.grid()
    file_name = "loss_curves.png"
    plt.savefig(file_name)
    print("-> Plot of loss curves are saved.")


# load data
mnist = input_data.read_data_sets('MNIST_data')

print("-" * 100)
print("batch_size = ", batch_size)
print("hidden_dim = ", hidden_dim)
print("epoch_number = ", epoch_number)
print("learning_rate = ", learning_rate)
print("-" * 100)

print("-> Session is started.")

# vae model
model = model()
x, y, train_mode, z, mu, sigma = model.encoder(hidden_dim)
decoder_output = model.decoder(z)
general_loss, rec_loss, kl_loss = model.loss(decoder_output, mu, sigma)
train_optimizer = model.optimizer(general_loss, learning_rate)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("-> Session is started.")

# main training loop
plot_general_loss_, plot_rec_loss_, plot_kl_loss_ = [], [], []
print("-> Training is started.")
for epoch in range(epoch_number):

    number_of_batch = int(mnist.train.num_examples / batch_size)

    general_loss_, rec_loss_, kl_loss_ = 0, 0, 0

    for i in range(number_of_batch):

        batch = mnist.train.next_batch(batch_size)[0]  # we take only data (not labels)
        batch = np.reshape(batch, [-1, 28, 28])  # reshape the data

        # train the network for one batch
        _, general_loss_, rec_loss_, kl_loss_, decoder_ = sess.run(
            [train_optimizer, general_loss, rec_loss, kl_loss, decoder_output],
            feed_dict={x: batch, y: batch, train_mode: True})

        if i == 0:
            print(
                "|--- Epoch: {} \t---> General loss: {:.2f} | Reconstruction loss: {:.2f} | KL-divergence loss: {:.2f}".format(
                    epoch, general_loss_, np.mean(rec_loss_), np.mean(kl_loss_)))

            plot_general_loss_.append(general_loss_)
            plot_rec_loss_.append(np.mean(rec_loss_))
            plot_kl_loss_.append(np.mean(kl_loss_))

plot_curves(plot_general_loss_, plot_rec_loss_, plot_kl_loss_, epoch_number)

# save model
saver = tf.train.Saver()
saver.save(sess, "model")
print("-> Model is saved.")

sess.close()

print("-> Session is ended.")
