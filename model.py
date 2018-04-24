import tensorflow as tf
import numpy as np

from tensorflow.contrib import layers

class Auto_Encoder():
	def __init__(self, trainable=False):

		self.data_height, self.data_width, self.data_channel = [1024, 32, 1]
		self.class_num = 4
		self.Z_dim = 192

		print("\n Building Model...")
		self.X = tf.placeholder(tf.float32, shape=[None, self.data_height, self.data_width, self.data_channel])
		self.Y = tf.placeholder(tf.float32, shape=[None, self.class_num])
		self.is_training = tf.placeholder(tf.bool, shape=None)
				
		self.Z = self.Encoder(self.X)
		self.X_hat = self.Decoder(self.Z)
		print(' Z:{0}, X_hat:{1}'.format(self.Z.shape, self.X_hat.shape))

		self.loss = tf.reduce_mean(tf.square(self.X - self.X_hat))
	
		if trainable:
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				self.train = tf.train.AdamOptimizer(0.001).minimize(self.loss)		


	def Encoder(self, X, reuse=False):
		with tf.variable_scope("Encoder") as scope:
			if reuse == True: scope.reuse_variables()

			Conv = layers.conv2d(inputs=X, num_outputs=64, kernel_size=5, stride=[2,1], activation_fn=tf.nn.relu) # 512
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d(inputs=Conv, num_outputs=64, kernel_size=5, stride=[2,1], activation_fn=tf.nn.relu) # 256
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))			
			Conv = layers.conv2d(inputs=Conv, num_outputs=96, kernel_size=5, stride=[2,1], activation_fn=tf.nn.relu) # 128
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d(inputs=Conv, num_outputs=96, kernel_size=5, stride=[2,1], activation_fn=tf.nn.relu) # 64
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d(inputs=Conv, num_outputs=128, kernel_size=5, stride=[2,1], activation_fn=tf.nn.relu) # 32
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d(inputs=Conv, num_outputs=128, kernel_size=4, stride=2, activation_fn=tf.nn.relu) # 16
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d(inputs=Conv, num_outputs=160, kernel_size=4, stride=2, activation_fn=tf.nn.relu) # 8
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d(inputs=Conv, num_outputs=160, kernel_size=4, stride=2, activation_fn=tf.nn.relu) # 4
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d(inputs=Conv, num_outputs=192, kernel_size=3, stride=2, activation_fn=tf.nn.relu) # 2
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d(inputs=Conv, num_outputs=self.Z_dim, kernel_size=3, stride=2, activation_fn=None)
			return Conv


	def Decoder(self, Z, reuse=False):
		with tf.variable_scope("Decoder") as scope:
			if reuse == True: scope.reuse_variables()

			Conv = layers.conv2d_transpose(inputs=Z, num_outputs=192, kernel_size=3, stride=2, activation_fn=tf.nn.relu) # 2
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d_transpose(inputs=Conv, num_outputs=160, kernel_size=3, stride=2, activation_fn=tf.nn.relu) # 4
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d_transpose(inputs=Conv, num_outputs=160, kernel_size=4, stride=2, activation_fn=tf.nn.relu) # 8
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d_transpose(inputs=Conv, num_outputs=128, kernel_size=4, stride=2, activation_fn=tf.nn.relu) # 16
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d_transpose(inputs=Conv, num_outputs=128, kernel_size=4, stride=2, activation_fn=tf.nn.relu) # 32
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d_transpose(inputs=Conv, num_outputs=96, kernel_size=5, stride=[2,1], activation_fn=tf.nn.relu) # 64
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d_transpose(inputs=Conv, num_outputs=96, kernel_size=5, stride=[2,1], activation_fn=tf.nn.relu) # 128
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d_transpose(inputs=Conv, num_outputs=64, kernel_size=5, stride=[2,1], activation_fn=tf.nn.relu) # 256
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d_transpose(inputs=Conv, num_outputs=64, kernel_size=5, stride=[2,1], activation_fn=tf.nn.relu) # 512
			Conv = tf.nn.relu(layers.batch_norm(inputs=Conv, is_training=self.is_training))
			Conv = layers.conv2d_transpose(inputs=Conv, num_outputs=1, kernel_size=5, stride=[2,1], activation_fn=None) # 1032		
			return Conv