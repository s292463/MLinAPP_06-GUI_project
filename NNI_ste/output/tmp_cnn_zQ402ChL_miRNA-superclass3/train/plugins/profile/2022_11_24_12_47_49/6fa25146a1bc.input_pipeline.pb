	z���3�@z���3�@!z���3�@	p�,2�� @p�,2�� @!p�,2�� @"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLz���3�@i;���n�?1i�'� @A�!S>�?IG�ŧ �@YDl�p���?rEagerKernelExecute 0*	��v���e@2F
Iterator::Modeluۈ'��?!��z�H@)Ҧ��\�?19�-�I�@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat4��7�¤?!W���Kv7@)(�r�w�?1��"h#�3@:Preprocessing2U
Iterator::Model::ParallelMapV2.Ȗ��2�?!,U3���/@).Ȗ��2�?1,U3���/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���B���?!{y� =qI@)�ؖg)�?1>=�>o,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate� �?!�.���'@)i�ai�G�?1�mR}@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�n��;��?!,A����@)�n��;��?1,A����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor$��(�[z?!VQ,A�@)$��(�[z?1VQ,A�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap5�b��^�?!P�Ii*@)��+��a?1	i�H�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�46.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9p�,2�� @Ib4KRyQ@QJ��<@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	i;���n�?i;���n�?!i;���n�?      ��!       "	i�'� @i�'� @!i�'� @*      ��!       2	�!S>�?�!S>�?!�!S>�?:	G�ŧ �@G�ŧ �@!G�ŧ �@B      ��!       J	Dl�p���?Dl�p���?!Dl�p���?R      ��!       Z	Dl�p���?Dl�p���?!Dl�p���?b      ��!       JGPUYp�,2�� @b qb4KRyQ@yJ��<@