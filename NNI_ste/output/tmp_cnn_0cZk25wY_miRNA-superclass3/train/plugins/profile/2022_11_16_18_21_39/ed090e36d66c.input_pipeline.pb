	P�c*�@P�c*�@!P�c*�@	�?>�{@�?>�{@!�?>�{@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLP�c*�@'��rJ �?1;6�>�?AK�|%��?I2���@Y�|?q �?rEagerKernelExecute 0*	ףp=
gs@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�AҧU��?!I�3�CTP@)(�I��?1�<=�<O@:Preprocessing2F
Iterator::ModelU������?!!ԑK�16@)H�`���?13�Ǭj/@:Preprocessing2U
Iterator::Model::ParallelMapV2Ü�M��?!���@)Ü�M��?1���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��5���?!����sS@)4�ތ���?1ιO�.�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceQN����?!�\�O�	@)QN����?1�\�O�	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��$���?!���J�@)��$���?1���J�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatel��3�I�?!�!צ�@)�Nϻ���?1=���,�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�ڧ�1�?!/��p@)�a0��e?1V����m�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�44.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�?>�{@Iyg����P@Q����y;@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	'��rJ �?'��rJ �?!'��rJ �?      ��!       "	;6�>�?;6�>�?!;6�>�?*      ��!       2	K�|%��?K�|%��?!K�|%��?:	2���@2���@!2���@B      ��!       J	�|?q �?�|?q �?!�|?q �?R      ��!       Z	�|?q �?�|?q �?!�|?q �?b      ��!       JGPUY�?>�{@b qyg����P@y����y;@