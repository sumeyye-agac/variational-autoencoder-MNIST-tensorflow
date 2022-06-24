import numpy as np
import matplotlib.pyplot as plt
from model import model
import tensorflow as tf


hidden_dim = 64

model = model()
_, _, train_mode, z, _, _ = model.encoder(hidden_dim)
decoder_output = model.decoder(z)

saver = tf.train.Saver()
sess = tf.Session()

saver.restore(sess, './model')
print("-> Checkpoints are restored from model")

randoms = [np.random.normal(0, 1, hidden_dim) for _ in range(100)]

print("-> Generation is started.")

images = sess.run(decoder_output, feed_dict = {z: randoms, train_mode: False})
images = [np.reshape(images[i], [28, 28]) for i in range(len(images))]

# https://medium.com/the-data-science-publication/how-to-plot-mnist-digits-using-matplotlib-65a2e0cc068
num_row, num_col = 10, 10
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col, 2*num_row))
for i in range(num_row*num_col):
  ax = axes[i//num_col, i%num_col]
  ax.imshow(images[i], cmap='gray')
  ax.set_title('image {}'.format(i))
plt.tight_layout()
file_name = "generated_images.png"
plt.savefig(file_name)
print("-> Generated images are saved in " + file_name)

sess.close()