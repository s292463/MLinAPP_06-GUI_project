	l� [�@l� [�@!l� [�@	m(�yp@m(�yp@!m(�yp@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLl� [�@ș&l?Y�?10� ��G@A���f�8�?I,,��@Y(|���?rEagerKernelExecute 0*	A`��"?b@2F
Iterator::Model��4c�t�?!����ӱH@)�w��Dg�?1�!"R��@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatnnLOX�?!�fh^��7@)P�Y��/�?1����3@:Preprocessing2U
Iterator::Model::ParallelMapV2Xc'��?!�O�1��.@)Xc'��?1�O�1��.@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicek�) Ɠ?!Ry4��t*@)k�) Ɠ?1Ry4��t*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate}"O����?!�'�8"3@)�n��\��?1��6.�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���)��?!a
`a,NI@)q�{��c�?1����nD@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor.v��2Sz?!Mۅ���@).v��2Sz?1Mۅ���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapͱ��0�?!ʆn�d�4@)}zlˀ�d?1
{j²�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 14.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�29.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9m(�yp@I�WvQ!EF@Q���o�lI@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ș&l?Y�?ș&l?Y�?!ș&l?Y�?      ��!       "	0� ��G@0� ��G@!0� ��G@*      ��!       2	���f�8�?���f�8�?!���f�8�?:	,,��@,,��@!,,��@B      ��!       J	(|���?(|���?!(|���?R      ��!       Z	(|���?(|���?!(|���?b      ��!       JGPUYm(�yp@b q�WvQ!EF@y���o�lI@