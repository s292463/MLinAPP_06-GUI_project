	�Q��Z�/@�Q��Z�/@!�Q��Z�/@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�Q��Z�/@XU/����?1��9��(@AA�! 8�?I�H���N�?rEagerKernelExecute 0*	��x�&Ye@2F
Iterator::Model��6���?!�0��"I@)����:�?15��$'�B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�%r���?!�Q~Y$9@)���:TS�?1fq�t�4@:Preprocessing2U
Iterator::Model::ParallelMapV2��7��?!��x�N*@)��7��?1��x�N*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateg�ba���?!ظƁ��1@)��:��?1ᒾ?a9$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�Z(��ډ?!�����@)�Z(��ډ?1�����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�f׽�?!Q��<&�H@)l��g���?1��ydE@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�p;4,F}?!f�&4�@)�p;4,F}?1f�&4�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaps��/٠?!OJݙD3@)�����h?1Rd9�%:�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 10.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIL�܊�5@Qm�H��S@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	XU/����?XU/����?!XU/����?      ��!       "	��9��(@��9��(@!��9��(@*      ��!       2	A�! 8�?A�! 8�?!A�! 8�?:	�H���N�?�H���N�?!�H���N�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qL�܊�5@ym�H��S@