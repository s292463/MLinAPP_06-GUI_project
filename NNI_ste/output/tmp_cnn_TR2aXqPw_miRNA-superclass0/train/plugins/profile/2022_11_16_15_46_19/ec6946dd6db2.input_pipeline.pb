	�*��p-@�*��p-@!�*��p-@	7��Ih�?7��Ih�?!7��Ih�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�*��p-@�T3k) �?1�G7¢�@AAc&Q/�?I���ڧ%@Y��2p@�?rEagerKernelExecute 0*	�O��n�d@2F
Iterator::Model��>+�?!�\��lcJ@)fٓ���?1��]1.C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�1^�?!�t�{��6@)�Q�y9�?1��,��g2@:Preprocessing2U
Iterator::Model::ParallelMapV2�$��}8�?!c3"H��,@)�$��}8�?1c3"H��,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����4)�?!�Y�0)@)����4)�?1�Y�0)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate}$%=��?!֡�qB2@)S[� ��?1DL��%�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipW@�ճ?!�+��G@)��#0�?1n��ÄE@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����{?!��W�?�@)����{?1��W�?�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���:��?!j��s��3@)ĕ�wF[e?1H��  l�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 9.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�72.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no97��Ih�?I�i:₩T@Q��Z���.@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�T3k) �?�T3k) �?!�T3k) �?      ��!       "	�G7¢�@�G7¢�@!�G7¢�@*      ��!       2	Ac&Q/�?Ac&Q/�?!Ac&Q/�?:	���ڧ%@���ڧ%@!���ڧ%@B      ��!       J	��2p@�?��2p@�?!��2p@�?R      ��!       Z	��2p@�?��2p@�?!��2p@�?b      ��!       JGPUY7��Ih�?b q�i:₩T@y��Z���.@