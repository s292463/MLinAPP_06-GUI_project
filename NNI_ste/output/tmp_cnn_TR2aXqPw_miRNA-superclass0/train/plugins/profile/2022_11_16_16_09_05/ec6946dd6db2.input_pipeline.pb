	d�1�@d�1�@!d�1�@	��7��_@��7��_@!��7��_@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLd�1�@�:���?1��fF?�@A��,z��?IϽ�K�;�?Y���(_��?rEagerKernelExecute 0*	���(\�d@2F
Iterator::ModelK�8��մ?!�5��UH@)���s���?1l��l�B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatƉ�v�?!T��l�=@)�=�
Y�?1j�8��8@:Preprocessing2U
Iterator::Model::ParallelMapV2�K7�A`�?!�Lٟ2�(@)�K7�A`�?1�Lٟ2�(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicec����?!��:��8'@)c����?1��:��8'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�J�8���?!T�E�M�I@)=�r�}ǀ?1�Ҙ@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��X���?!s�'��/@)��9D�|?1O_ڽz�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor[a�^Cp|?!����h�@)[a�^Cp|?1����h�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��(	���?!��^��X1@)NA~6r�d?1v%��k^�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 16.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�26.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��7��_@Iؙ�7,E@Q��J5ɍK@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�:���?�:���?!�:���?      ��!       "	��fF?�@��fF?�@!��fF?�@*      ��!       2	��,z��?��,z��?!��,z��?:	Ͻ�K�;�?Ͻ�K�;�?!Ͻ�K�;�?B      ��!       J	���(_��?���(_��?!���(_��?R      ��!       Z	���(_��?���(_��?!���(_��?b      ��!       JGPUY��7��_@b qؙ�7,E@y��J5ɍK@