	`�+��V%@`�+��V%@!`�+��V%@	y��hO@y��hO@!y��hO@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL`�+��V%@��2n�	@1l�<�@A��^f�?In3���?Y�C?�{�?rEagerKernelExecute 0*	+��a@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenately�z�L�?!��.uy7D@)6��x"��?1>�T�B@:Preprocessing2F
Iterator::Modelm��]٭?!A7pl�D@)%"���1�?1��d�|:@:Preprocessing2U
Iterator::Model::ParallelMapV2ٕ��zO�?!;W�ch-@)ٕ��zO�?1;W�ch-@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�Գ ���?!�i5]F�(@)����y7�?1J�}!��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�-�\o{?!�)��@)�-�\o{?1�)��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipt���N�?!��ȏ�gM@)g��j+�w?1;��2{�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���7��?!�'��#E@)yx��ee?1���#��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicec�J!�K\?!����?)c�J!�K\?1����?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorc�J!�K\?!����?)c�J!�K\?1����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 30.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�18.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9y��hO@IHwL	]H@Q�H3 nH@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��2n�	@��2n�	@!��2n�	@      ��!       "	l�<�@l�<�@!l�<�@*      ��!       2	��^f�?��^f�?!��^f�?:	n3���?n3���?!n3���?B      ��!       J	�C?�{�?�C?�{�?!�C?�{�?R      ��!       Z	�C?�{�?�C?�{�?!�C?�{�?b      ��!       JGPUYy��hO@b qHwL	]H@y�H3 nH@