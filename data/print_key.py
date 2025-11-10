import tensorflow as tf

path = "data/block_push_visual_location/oracle_0.tfrecord"

raw_dataset = tf.data.TFRecordDataset(path)

# 取一条样本看看结构
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print("=== Keys in TFRecord ===")
    for k, v in example.features.feature.items():
        print(f"{k}: {len(v.bytes_list.value) or len(v.float_list.value) or len(v.int64_list.value)} values")
