	�����7@�����7@!�����7@	*ʎ��@*ʎ��@!*ʎ��@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0�����7@�b�� @1�*�C3O@Id�]K�#2@Y�r߉Y�?r0*	�G�z�w@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�	��b��?!�x(�A�K@)�	��b��?1�x(�A�K@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatj���'�?!i�'���3@)��W���?1��2��1@:Preprocessing2U
Iterator::Model::ParallelMapV2�g@�5�?!�s�q7 @)�g@�5�?1�s�q7 @:Preprocessing2F
Iterator::Model�3/��w�?!���.>-@)��Q���?1D5z�m@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapkg{��?!�KI
�TN@).��?1���"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���;���?!0n,<XU@)y�[Y��?1�5݃+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorr�)���?!�f�/�@)r�)���?1�f�/�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 8.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�77.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9*ʎ��@I�"��lU@Q]8kJ�%@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�b�� @�b�� @!�b�� @      ��!       "	�*�C3O@�*�C3O@!�*�C3O@*      ��!       2      ��!       :	d�]K�#2@d�]K�#2@!d�]K�#2@B      ��!       J	�r߉Y�?�r߉Y�?!�r߉Y�?R      ��!       Z	�r߉Y�?�r߉Y�?!�r߉Y�?b      ��!       JGPUY*ʎ��@b q�"��lU@y]8kJ�%@