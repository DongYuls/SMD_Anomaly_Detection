import argparse
def str2bool(str): 
	return str.lower() in ('true', '1')

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store', type=str2bool, dest='training', default=False)
config, unparsed = parser.parse_known_args()

import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # to ignore tensorflow SSE instruction warning
import numpy as np
import tensorflow as tf

from model import *
model = Auto_Encoder(trainable=config.training)
from setup_dataset import *

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

ckpt_path = os.path.join(os.getcwd(), 'checkpoint')

def training_model():
	training_steps = 10000
	training_batch_size = 32
	
	training_loss = tf.get_variable(name='Training-loss', shape=[], initializer=tf.zeros_initializer())
	avg_normal_loss = tf.get_variable(name='Normal-loss', shape=[], initializer=tf.zeros_initializer())
	avg_abnormal_loss = tf.get_variable(name='Abnormal-loss', shape=[], initializer=tf.zeros_initializer())

	tf.summary.scalar(training_loss.name, training_loss)
	tf.summary.scalar(avg_normal_loss.name, avg_normal_loss)
	tf.summary.scalar(avg_abnormal_loss.name, avg_abnormal_loss)
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter(ckpt_path)

	saver = tf.train.Saver()
	
	dataset = TFRecord(values=[model.X, model.Y], batch_size=33)
	sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	for step in range(training_steps):
		X_data, Y_data = sess.run(dataset.train.mini_batch) 
		sess.run(model.train, feed_dict={model.X: X_data, model.Y: Y_data, model.is_training: True})
		sys.stdout.write(" Step: %4d ...\r" % step)

		if step % 100 == 0:
			normal_loss, abnormal_loss = [], []
			loss_curr = sess.run(model.loss, feed_dict={model.X: X_data, model.Y: Y_data, model.is_training: True})

			for i in range(64): # test set length: 2,113 / test batch size: 33	
				X_data, Y_data = sess.run(dataset.test.mini_batch)
				Normal_X, Abnormal_X = X_data[Y_data.T[3]==1], X_data[Y_data.T[3]!=1] # normal index=3
				if len(Normal_X) > 0:
					normal_loss.append(sess.run(model.loss, feed_dict={model.X: Normal_X, model.is_training: False}))
				if len(Abnormal_X) > 0:
					abnormal_loss.append(sess.run(model.loss, feed_dict={model.X: Abnormal_X, model.is_training: False}))

			sess.run(avg_normal_loss.assign(np.mean(normal_loss)))
			sess.run(avg_abnormal_loss.assign(np.mean(abnormal_loss)))
			sess.run(training_loss.assign(loss_curr))
			writer.add_summary(sess.run(merged_summary), step)
		
			print(' Step: {0:>4}   Training_loss: {1:>0.5f}   Test_loss(Normal, Abnormal): {2:>0.4f}, {3:>0.4f}\r'.\
					format(step, loss_curr, np.mean(normal_loss), np.mean(abnormal_loss)))

	print("\n Saving Model Complete...")
	saver.save(sess, os.path.join(ckpt_path, 'model.ckpt'))
	
	coord.request_stop()
	coord.join(threads)
	sess.close()


def validation():
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	import matplotlib.patches as mpatches

	dataset = TFRecord(values=[model.X, model.Y], batch_size=1, test_mode=True)
	sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

	saver = tf.train.Saver()
	saver.restore(sess, os.path.join(ckpt_path, 'model.ckpt'))

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	abnormal_score, normal_score = [[],[],[]], []

	input_count = 0
	# check residual error with single sample 
	while not coord.should_stop():
		try: X_data, Y_data = sess.run(dataset.test.mini_batch)
		except tf.errors.OutOfRangeError: break
		input_count += 1

		label = np.argmax(Y_data[0], axis=0)
		score = sess.run(model.loss, feed_dict={model.X: X_data, model.Y: Y_data, model.is_training: False})
		
		if label == 3: normal_score.append(score)
		else: abnormal_score[label].append(score)
		print('[{0:>2}] {1} :: {2:>0.4f}'.format(input_count, label, score))
	
	coord.request_stop()
	coord.join(threads)

	# ensemble (the number of samples used for anomaly determinant)
	seg_num_for_ensemble = 1 # 1 ~ 20

	avg_abnormal, avg_normal = [[],[],[]], []
	while len(normal_score) > 0:
		avg_normal.append(np.mean(normal_score[:seg_num_for_ensemble]))
		normal_score = normal_score[seg_num_for_ensemble:]
	
	for i in range(3):
		while len(abnormal_score[i]) > 0:
			avg_abnormal[i].append(np.mean(abnormal_score[i][:seg_num_for_ensemble]))
			abnormal_score[i] = abnormal_score[i][seg_num_for_ensemble:]
	
	color = ['green', 'darkorange', 'cornflowerblue', 'deeppink']

	# draw roc curve for each class
	import sklearn.metrics
	import itertools
	from scipy import interp as scipy_interp

	label_list = ['B-line', 'C-line with intermittent noise', 'Non-greased C-line']
	for i in range(3):
		false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve( \
			np.concatenate((np.zeros(len(avg_normal)), np.ones(len(avg_abnormal[i]))), axis=0), \
			np.concatenate((np.asarray(avg_normal), np.asarray(avg_abnormal[i])), axis=0))

		ROC_AUC = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

		plt.plot(false_pos_rate, true_pos_rate, color=color[i+1], lw=2, \
			label='%s (AUC = %0.2f)' % (label_list[i], ROC_AUC))
	
	plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
	plt.xlabel('1 - Specificity'); plt.ylabel('Sensitivity')
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc="lower right")
	plt.savefig('./ROC_AUC.png'); plt.clf()

	sess.close()
	
if __name__ == '__main__':	
	if(config.training==True): training_model()
	else: validation()