	^��-�I)@^��-�I)@!^��-�I)@	+~�jM@+~�jM@!+~�jM@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL^��-�I)@��[1�?1S�1�#�@A`L8��?I�>�G��@Y8��n��?rEagerKernelExecute 0*	T㥛��q@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��#0��?!�E�OQ@)Yک��`�?1f����P@:Preprocessing2U
Iterator::Model::ParallelMapV2�+I����?!�c�H'@)�+I����?1�c�H'@:Preprocessing2F
Iterator::Modelk+��ݓ�?!���Nm�5@)P�Y��/�?1�]U:�$@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�hE,�?!F�o0�@)��*�w��?1a�	�.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipD�r�c��?!ГS���S@)��K�[�?1I�#��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor|DL�$zy?!j.���v@)|DL�$zy?1j.���v@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�lw�N�?!yJ\YQ@)�����g?1��
/�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor�KTole?!{����^�?)�KTole?1{����^�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceςP���\?!���Ji��?)ςP���\?1���Ji��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 13.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�42.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9+~�jM@I�0�7.yL@Q8G(�aD@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��[1�?��[1�?!��[1�?      ��!       "	S�1�#�@S�1�#�@!S�1�#�@*      ��!       2	`L8��?`L8��?!`L8��?:	�>�G��@�>�G��@!�>�G��@B      ��!       J	8��n��?8��n��?!8��n��?R      ��!       Z	8��n��?8��n��?!8��n��?b      ��!       JGPUY+~�jM@b q�0�7.yL@y8G(�aD@