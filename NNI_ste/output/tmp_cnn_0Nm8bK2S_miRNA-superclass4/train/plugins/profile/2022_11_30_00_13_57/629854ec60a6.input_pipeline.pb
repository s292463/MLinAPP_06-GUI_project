	�<�r@�<�r@!�<�r@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�<�r@��E;��?1N�W�z@AC�l搔?I�z�ю�	@rEagerKernelExecute 0*	��K7��f@2F
Iterator::Model�P�[��?!����bG@)��IӠh�?1�q~((@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�����?!1ɠKi�;@)�!H��?1�Pg`�8@:Preprocessing2U
Iterator::Model::ParallelMapV2m�i�*��?!�K���+@)m�i�*��?1�K���+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice)!XU/��?!1,�G��$@))!XU/��?11,�G��$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipD�R�Z�?!|{a��J@)1ҋ��*�?1B��J.!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�D���?!���O-�/@)��~��΃?1i5a�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorj��_=�{?!���Y/�@)j��_=�{?1���Y/�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��H����?!%D�j��1@)�c�3�%k?1)�)\��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 7.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�47.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIuVQq�\K@Q����e�F@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��E;��?��E;��?!��E;��?      ��!       "	N�W�z@N�W�z@!N�W�z@*      ��!       2	C�l搔?C�l搔?!C�l搔?:	�z�ю�	@�z�ю�	@!�z�ю�	@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb quVQq�\K@y����e�F@