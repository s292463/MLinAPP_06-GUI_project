	|Bv��@|Bv��@!|Bv��@	c3�֣�@c3�֣�@!c3�֣�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL|Bv��@B�/h!A�?1��
@A1^�Κ?I��X32H�?Ya7l[���?rEagerKernelExecute 0*	�x�&1b@2F
Iterator::Model^���?!_��$]�I@)����ө?1�UW~lA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���8��?!Z����;@)�rK�!�?10�K�`7@:Preprocessing2U
Iterator::Model::ParallelMapV2�V�����?!������0@)�V�����?1������0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice~�*O �?!�5�C�:@)~�*O �?1�5�C�:@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�����?!�fdۢ:H@)�@�"�?1�i�Ek@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�شR�?!m��O})@)��>V�ۀ?19��[K�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorpD��k�|?!wC˪p]@)pD��k�|?1wC˪p]@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap ��WW�?!��M�n�,@)2��|�c?1����4s�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 16.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�21.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9c3�֣�@I��8} �B@Q�ZE M@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	B�/h!A�?B�/h!A�?!B�/h!A�?      ��!       "	��
@��
@!��
@*      ��!       2	1^�Κ?1^�Κ?!1^�Κ?:	��X32H�?��X32H�?!��X32H�?B      ��!       J	a7l[���?a7l[���?!a7l[���?R      ��!       Z	a7l[���?a7l[���?!a7l[���?b      ��!       JGPUYc3�֣�@b q��8} �B@y�ZE M@