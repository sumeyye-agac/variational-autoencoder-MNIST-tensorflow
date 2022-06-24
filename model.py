import tensorflow as tf

class model():

  def __init__(self):
    self.x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='x')
    self.y = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='y')
    self.train_mode = tf.placeholder(tf.bool, name="train_mode")

  def encoder(self, hidden_dim):
    print("----- Encoder -----")
    print("input hidden_dim: ", hidden_dim)
    print("input x: ", self.x.shape)

    lstm, _ = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(hidden_dim), self.x, dtype=tf.float32)
    print("lstm: ", lstm.shape)
    flatten_lstm = tf.contrib.layers.flatten(lstm)
    print("flatten_lstm: ", flatten_lstm.shape)

    mu = tf.layers.dense(flatten_lstm, units=hidden_dim, activation=tf.nn.relu)
    sigma = 0.5 * tf.layers.dense(flatten_lstm, units=hidden_dim)
    epsilon = tf.random_normal(tf.stack([tf.shape(flatten_lstm)[0], hidden_dim]))
    z = mu + tf.multiply(epsilon, tf.exp(sigma))   
    print("mu: ", mu.shape)
    print("sigma: ", sigma.shape)
    print("epsilon: ", epsilon.shape)
    print("z: ", z.shape)

    return self.x, self.y, self.train_mode, z, mu, sigma


  def decoder(self, z):
    print("----- Decoder -----")
    print("input z: ", z.shape)
    reshaped_dimension = [-1, 14, 14, 1]

    fc1 = tf.layers.dense(z, units=98, activation=tf.nn.relu)
    print("1. fc: ", fc1.shape)
    fc2 = tf.layers.dense(fc1, units=196, activation=tf.nn.relu)
    print("2. fc: ", fc2.shape)

    reshaped_fc2 = tf.reshape(fc2, reshaped_dimension)
    print("2. reshaped_fc: ", reshaped_fc2.shape)

    tconv1 = tf.layers.conv2d_transpose(reshaped_fc2, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(tconv1, rate=0.75, training=self.train_mode)
    print("3. tconv: ", tconv1.shape)
    print("3. dropout: ", dropout1.shape)

    tconv2 = tf.layers.conv2d_transpose(dropout1, filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(tconv2, rate=0.75, training=self.train_mode)
    print("4. tconv: ", tconv2.shape)
    print("4. dropout: ", dropout2.shape)

    tconv3 = tf.layers.conv2d_transpose(dropout2, filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
    #dropout3 = tf.layers.dropout(tconv3, rate=0.75, training=self.train_mode)
    print("5. tconv: ", tconv3.shape)
    #print("5. dropout: ", dropout3.shape)

    flatten_tconv3 = tf.contrib.layers.flatten(tconv3)
    print("5. flatten_tconv: ", flatten_tconv3.shape)

    fc3 = tf.layers.dense(flatten_tconv3, units=28*28, activation=tf.nn.sigmoid)
    print("6. fc: ", fc3.shape)

    reshaped_output = tf.reshape(fc3, shape=[-1, 28, 28])
    print("6. reshaped_output: ", reshaped_output.shape)

    return reshaped_output


  def loss(self, decoder, mu, sigma):
    flatten_y = tf.reshape(self.y, shape=[-1, 28 * 28])
    unreshaped = tf.reshape(decoder, [-1, 28 * 28])
    unreshaped = tf.clip_by_value(unreshaped, 1e-7, 1-1e-7) # clipping is added for bce loss problems

    # binary cross entropy loss (reconstruction loss)
    bce_loss = tf.reduce_sum(-flatten_y*tf.log(unreshaped)-(1.0 - flatten_y)*tf.log(1.0 - unreshaped), axis=1)

    # kl divergence loss:
    kl_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sigma - tf.square(mu) - tf.exp(2.0 * sigma), axis=1)

    # general loss
    general_loss = tf.reduce_mean(bce_loss + kl_loss, axis=0)

    return general_loss, bce_loss, kl_loss


  def optimizer(self, loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return optimizer
