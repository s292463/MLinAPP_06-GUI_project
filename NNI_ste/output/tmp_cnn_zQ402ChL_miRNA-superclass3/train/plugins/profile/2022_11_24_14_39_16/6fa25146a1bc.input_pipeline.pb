	Ή=��%@Ή=��%@!Ή=��%@	��J��@��J��@!��J��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLΉ=��%@]�&��?1Y�|^��?A>yX�5ͫ?I���{G�@Y���/g�?rEagerKernelExecute 0*	�S㥛�t@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���"��?!{��8��N@)������?1J�B5mK@:Preprocessing2F
Iterator::Model��MbX�?!^f��'�4@)_EF$a�?1bFsN��+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�����?!�k⁠',@)��w��x�?1�¬)'@:Preprocessing2U
Iterator::Model::ParallelMapV2	��z���?!��=��@)	��z���?1��=��@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����(y�?!�*��@)����(y�?1�*��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipn�ݳ��?!i�U��S@)���k��?1�wP�>�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor� [��ˀ?!DT/�@)� [��ˀ?1DT/�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapnQf�L2�?!d*��(O@)��bg
m?1]�\E�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�49.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t21.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9��J��@I]t���Q@Q�@��6@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	]�&��?]�&��?!]�&��?      ��!       "	Y�|^��?Y�|^��?!Y�|^��?*      ��!       2	>yX�5ͫ?>yX�5ͫ?!>yX�5ͫ?:	���{G�@���{G�@!���{G�@B      ��!       J	���/g�?���/g�?!���/g�?R      ��!       Z	���/g�?���/g�?!���/g�?b      ��!       JGPUY��J��@b q]t���Q@y�@��6@