	u/3l�@u/3l�@!u/3l�@	Y�:��@Y�:��@!Y�:��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLu/3l�@x'���?1�t"�T�?A�t �՗?IY�O0^@Y��bg
�?rEagerKernelExecute 0*	�����t@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��Tƿ�?!&�b$L@)�s]�@�?1R����I@:Preprocessing2F
Iterator::Model�ʆ5�E�?!�O(�!9@)��0휮?1�/$��2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�@,�9$�?!.��q�(@)>�h�?1�Uf�]�$@:Preprocessing2U
Iterator::Model::ParallelMapV2Ӿ��zܗ?!#��h1@)Ӿ��zܗ?1#��h1@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�
����?!��'��@)�
����?1��'��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipW�c#��?!H8�5��R@)2t�ב?1U<ݝ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorI���*�}?!��pP�@)I���*�}?1��pP�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�{h+�?!�&����L@)}�E�j?1��G��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�54.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Y�:��@I�P=ʌS@Q>m�2@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	x'���?x'���?!x'���?      ��!       "	�t"�T�?�t"�T�?!�t"�T�?*      ��!       2	�t �՗?�t �՗?!�t �՗?:	Y�O0^@Y�O0^@!Y�O0^@B      ��!       J	��bg
�?��bg
�?!��bg
�?R      ��!       Z	��bg
�?��bg
�?!��bg
�?b      ��!       JGPUYY�:��@b q�P=ʌS@y>m�2@