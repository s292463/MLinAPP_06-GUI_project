	��J
,@��J
,@!��J
,@	���z:	@���z:	@!���z:	@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��J
,@�wD���?1\��J��@A#��~j��?IG�����?Y ����m�?rEagerKernelExecute 0*	t�V�a@2F
Iterator::ModelʉvR�?!qT���I@)�&�W�?1��i�]B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatOt]����?!�K%c?=@)%xC8�?1�Co�v8@:Preprocessing2U
Iterator::Model::ParallelMapV2bۢ���?!�Q��,@)bۢ���?1�Q��,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceP�<��?!��>%�f@)P�<��?1��>%�f@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��j�j��?!"�4H�)@)31]��?1��C�1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�������?!���StH@)�)"�*~?1���)@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorT���f~?!�@Ա�@)T���f~?1�@Ա�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���T���?!�ͩ�p�,@)E+��Ba?1�WpD�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 16.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�20.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���z:	@IS�jHZ�B@QC�y��M@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�wD���?�wD���?!�wD���?      ��!       "	\��J��@\��J��@!\��J��@*      ��!       2	#��~j��?#��~j��?!#��~j��?:	G�����?G�����?!G�����?B      ��!       J	 ����m�? ����m�?! ����m�?R      ��!       Z	 ����m�? ����m�?! ����m�?b      ��!       JGPUY���z:	@b qS�jHZ�B@yC�y��M@