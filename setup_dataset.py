import tensorflow as tf
import numpy as np
import os, sys, glob, time
import psutil
process = psutil.Process(os.getpid())

AUDIO_DATA_PATH = os.path.join(os.getcwd(), 'data')
TFRECORD_PATH = os.path.join(os.getcwd(), 'tfrecords')

""" Directory Info.
- data
	- _Abnormal_B_Line
	- _Abnormal_C_Line_(Noise)
	- _Abnormal_C_Line_(Non-greased)
	- _Normal_C_Line
	- _Normal_C_Line_(Test)
- tfrecords
	- test_set_00.tfrecords.gz
	- training_set_00.tfrecords.gz
"""

class Dataset():
	def __init__(self):
		print('\n Loading Datasets...')		
		self.data_height, self.data_width, self.data_channel = [1024, 32, 1]
		self.class_num = 4 # abnormal class: 0,1,2 / normal class: 3,4(test)

    # only normal data is used for training set, and the others are test set.
	def load_data(self, normal_index=3):
		import librosa

		file_list = os.listdir(AUDIO_DATA_PATH)
		file_list.sort()
		
		print('\n Labels:')
		for i, label in enumerate(file_list):
			print('   ({0}): {1}'.format(i, label))
			if i == len(file_list)-1: print('')

		test_audio, test_labels = list(), list()
		
		for i, label in enumerate(file_list):
			audio = []
			for file in glob.glob(os.path.join(AUDIO_DATA_PATH, label, '*.wav')):
				sys.stdout.write(' Extracting {0}...'.format(os.path.split(file)[-1]))		
				
				raw, sr = librosa.load(file, sr=16300, dtype=np.float32)
				data = abs(librosa.stft(raw, n_fft=2046, hop_length=512)) # STFT (window_size=2048, overlap_length=512)
				
				while True: # segmentation
					try: segment = np.reshape(data[:, :self.data_width], \
										[self.data_height, self.data_width, self.data_channel])
					except: break
					audio.append((segment-np.mean(segment))/np.std(segment))
					data = data[:, self.data_width//3:] # monitoring overlap ratio = 1/3
				
				print(' {0}'.format(len(audio)))
				print(' -- Memory Usage: {0:>0.3f}'.format(process.memory_percent()))

			labels = np.zeros((len(audio), self.class_num), dtype=np.int64)
			labels[np.arange(0, len(audio)), i if i<=3 else 3] = 1
			
			if i == normal_index: # training set: index=3 (normal)
				self.train = Sub_Dataset(np.asarray(audio), labels)
			else: # test set: index=0,1,2 (abnormal), 4 (normal test)
				test_audio.extend(audio)
				test_labels.extend(labels)
		self.test = Sub_Dataset(np.asarray(test_audio), np.asarray(test_labels))

		print(' Total Amount: Train({0}), Test({1})'.format(len(self.train.data), len(self.test.data)))


class Sub_Dataset():
	def __init__(self, data, labels, shuffle=True):
		self.data, self.labels = data, labels
		if shuffle: self.shuffle()
		self.batch_index = 0

	def next_batch(self, batch_size):
		next_index = self.batch_index + batch_size
		if(next_index < len(self.data)):
			batch_data, batch_label = self.data[self.batch_index:next_index], self.labels[self.batch_index:next_index]
			self.batch_index = next_index
		else:
			batch_data, batch_label = self.data[self.batch_index:], self.labels[self.batch_index:]
			self.shuffle()
			self.batch_index = 0

		return batch_data, batch_label

	def shuffle(self):
		assert len(self.data) == len(self.labels)
		p = np.random.permutation(len(self.data))
		self.data = self.data[p]
		self.labels = self.labels[p]


class TFRecord_Saver():
	def __init__(self, save_path):
		self.save_path = save_path
		if not os.path.exists(self.save_path):
			os.mkdir(self.save_path)

	def _float_feature(self, value): 
		return tf.train.Feature(float_list=tf.train.FloatList(value=value))

	def _bytes_feature(self, value): 
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

	def _int64_feature(self, value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
	
	def save_tfrecord(self, dataset, tfrec_name):
		print('\n TFRecord Writer Info')
		print(' - Data Length: {0}'.format(len(dataset.data)))
		print(' - Compression Type: GZIP (Option.{0})\n'.format(tf.python_io.TFRecordCompressionType.GZIP))
		
		file_num = 0
		save_path = os.path.join(self.save_path, tfrec_name+'_set_%02d.tfrecords.gz' % file_num)
		
		while os.path.exists(save_path):
			file_num += 1
			save_path = os.path.join(TFRECORD_PATH, tfrec_name+'_set_%02d.tfrecords.gz' % file_num)

		print(' Saving Dataset...({0}_set_{1:02})'.format(tfrec_name, file_num))
		std_time = time.time()
		writer = tf.python_io.TFRecordWriter(save_path, tf.python_io.TFRecordOptions(2))

		for i in range(len(dataset.data)):
			features = {'data': self._float_feature(dataset.data[i].flatten()),
						'labels': self._int64_feature(dataset.labels[i])}
			example = tf.train.Example(features=tf.train.Features(feature=features))	
			writer.write(example.SerializeToString())	
		
		writer.close()
		print(' -- {0:0.4f}\n'.format(float(time.time()-std_time)))


class TFRecord_Reader():
	def __init__(self, values, tfrec_name, batch_size, num_epochs=None, shuffle=None):
		self.data_shape = values[0].shape[1:].as_list()
		self.class_num = values[1].shape[1:].as_list()[0]

		if num_epochs == 1: self.allow_smaller_final_batch = True
		else: self.allow_smaller_final_batch = False

		filenames = [file for file in glob.glob(os.path.join(TFRECORD_PATH, tfrec_name+'*.tfrecords.gz'))]
		assert len(filenames) > 0, "No TFRecord Files"

		self.filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
		self.reader = tf.TFRecordReader(options=tf.python_io.TFRecordOptions(2))

		if shuffle: self.mini_batch = self.next_shuffle_batch(batch_size)
		else: self.mini_batch = self.next_batch(batch_size)


	def next_batch(self, batch_size):
		data, label = self.read_tfrecord()
		return tf.train.batch([data, label], batch_size=batch_size, capacity=batch_size*4, \
								num_threads=5, allow_smaller_final_batch=self.allow_smaller_final_batch)


	def next_shuffle_batch(self, batch_size):
		data, label = self.read_tfrecord()
		return tf.train.shuffle_batch([data, label], batch_size=batch_size, capacity=batch_size*4, num_threads=5, \
								min_after_dequeue=batch_size*2, allow_smaller_final_batch=self.allow_smaller_final_batch)


	def read_tfrecord(self):
		key, serialized_example = self.reader.read(self.filename_queue)
		features = {'data': tf.FixedLenFeature([np.prod(self.data_shape)], tf.float32), \
					'labels': tf.FixedLenFeature([self.class_num], tf.int64)}
		example = tf.parse_single_example(serialized_example, features=features)		
		return tf.reshape(example['data'], self.data_shape), example['labels']


class TFRecord():
	def __init__(self, values, batch_size, test_mode=False):
		print("\n Loading TFRecords...\n")
		num_epochs = None
		self.train = TFRecord_Reader(values=values, tfrec_name='training', \
						batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
		if test_mode: num_epochs = 1
		self.test = TFRecord_Reader(values=values, tfrec_name='test', \
						batch_size=batch_size, num_epochs=num_epochs, shuffle=False)


if __name__ == "__main__":
	
	dataset = Dataset()

	if len(sys.argv) == 1:
		if len(os.listdir(TFRECORD_PATH)) < 1:
			dataset.load_data()
			tfrecord = TFRecord_Saver(save_path=TFRECORD_PATH)
			tfrecord.save_tfrecord(dataset=dataset.train, tfrec_name='training')
			tfrecord.save_tfrecord(dataset=dataset.test, tfrec_name='test')

	else: # usage: python setup_dataset_v2.py test

		sess = tf.Session()

		X = tf.placeholder(tf.float32, shape=[None]+[dataset.data_height, dataset.data_width, dataset.data_channel])
		Y = tf.placeholder(tf.float32, shape=[None, dataset.class_num])

		tfrecord = TFRecord(values=[X, Y], batch_size=1, test_mode=True)
		
		sess.run(tf.group(tf.local_variables_initializer(), tf.global_variables_initializer()))
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		data_length = {'TOTAL': 0, '0':0, '1':0, '2':0, '3':0}

		with coord.stop_on_exception():
			while not coord.should_stop():
				X_data, Y_data = sess.run(tfrecord.test.mini_batch)
				label = np.argmax(Y_data[0])
				data_length[str(label)] += 1
				data_length['TOTAL'] += 1

				print(' X:{0} Y:{1} -- Total Length:{2:>4d} (0:{3:>4d}, 1:{4:>4d}, 2:{5:>4d}, 3:{6:>4d})'.\
					format(X_data.shape, Y_data.shape, data_length['TOTAL'], \
						data_length['0'], data_length['1'], data_length['2'], data_length['3']))

		coord.request_stop()
		coord.join(threads)
		sess.close()