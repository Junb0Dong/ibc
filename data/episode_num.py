import tensorflow as tf
import glob

tfrecord_files = sorted(glob.glob("/home/ps/ibc/data/test_dataset/*oracle_*.tfrecord"))
raw_dataset = tf.data.TFRecordDataset(tfrecord_files)

count = 0
for record in raw_dataset:
    count += 1
print("TFRecord 中 example 数量:", count)
