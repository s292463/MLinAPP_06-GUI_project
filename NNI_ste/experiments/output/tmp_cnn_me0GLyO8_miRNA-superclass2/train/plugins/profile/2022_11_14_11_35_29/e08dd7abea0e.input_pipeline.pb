	 Uܸ�t!@ Uܸ�t!@! Uܸ�t!@	ԢI��@ԢI��@!ԢI��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL Uܸ�t!@���а��?1Q�?Û�?A/�HM��?I�A_z�3@Y,���o�?rEagerKernelExecute 0*	�&1�0e@2F
Iterator::Model�>V���?!��s�E@)���H.�?1��%�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����[�?!T�+&��:@)]�gA(�?10�����6@:Preprocessing2U
Iterator::Model::ParallelMapV2w�
���?!e	#��91@)w�
���?1e	#��91@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice~��7L�?!nt��b'@)~��7L�?1nt��b'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�S�����?![��ǰ	7@)X�%����?1������&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipM��~�T�?!U7��L@)���u6�?1R79 ,_@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?���e{?!*�_�@)?���e{?1*�_�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�)�TPQ�?!�)����8@)>]ݱ�&e?1Yr���^�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 21.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�57.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ӢI��@I�9��Z�S@Q�����0@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���а��?���а��?!���а��?      ��!       "	Q�?Û�?Q�?Û�?!Q�?Û�?*      ��!       2	/�HM��?/�HM��?!/�HM��?:	�A_z�3@�A_z�3@!�A_z�3@B      ��!       J	,���o�?,���o�?!,���o�?R      ��!       Z	,���o�?,���o�?!,���o�?b      ��!       JGPUYӢI��@b q�9��Z�S@y�����0@