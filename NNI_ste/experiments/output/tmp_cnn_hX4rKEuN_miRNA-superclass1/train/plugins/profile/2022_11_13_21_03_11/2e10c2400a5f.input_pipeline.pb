	���j�@���j�@!���j�@	2�,�9@2�,�9@!2�,�9@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL���j�@�����?1ްmQfc@A�뤾,�?I-$`ty� @Yg�ba���?rEagerKernelExecute 0*	��C�b@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeath��?!w��&�6B@)Z)r�#�?1+��leZ?@:Preprocessing2F
Iterator::Model�\����?!�p�D@)�4c�tv�?1b*|�Y9@:Preprocessing2U
Iterator::Model::ParallelMapV2�O��5�?!���DV.@)�O��5�?1���DV.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatehB�Ēr�?!��}�/@)�'�>��?1DW�A�P!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�\QJV�?!�皪�@)�\QJV�?1�皪�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip �����?!�j�x��M@)�^(`;�?1K��)@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorp�^}<�}?!˿�;K@)p�^}<�}?1˿�;K@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM����'�?!*)�S�1@)/���ިe?1��aZMY�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 8.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�40.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no92�,�9@Iz��i�H@Q`{�csG@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�����?�����?!�����?      ��!       "	ްmQfc@ްmQfc@!ްmQfc@*      ��!       2	�뤾,�?�뤾,�?!�뤾,�?:	-$`ty� @-$`ty� @!-$`ty� @B      ��!       J	g�ba���?g�ba���?!g�ba���?R      ��!       Z	g�ba���?g�ba���?!g�ba���?b      ��!       JGPUY2�,�9@b qz��i�H@y`{�csG@