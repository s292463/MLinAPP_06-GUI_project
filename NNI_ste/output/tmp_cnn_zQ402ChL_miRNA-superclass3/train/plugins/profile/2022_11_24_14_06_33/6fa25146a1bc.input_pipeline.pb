	!?�n: @!?�n: @!!?�n: @	��n�?��n�?!��n�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL!?�n: @�=Ab��?1����( �?A�����?I��6p
@Yf��@�9�?rEagerKernelExecute 0*	X9���r@2F
Iterator::Model�E�����?!KdF��R@)���P�v�?1;�6�>�O@:Preprocessing2U
Iterator::Model::ParallelMapV2����x�?!neW]U�&@)����x�?1neW]U�&@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX����?![V/g�x*@)$�F��?1�����%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�f�R@ڳ?!�n�X�9@)~t��gy�?1��71�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceF�-t%�?!\��@)F�-t%�?1\��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���#bJ�?!U����,@)�=�Ӟ��?1L�Q��?	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��h:;|?!�>z��@)��h:;|?1�>z��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�Y�X"�?!�Y<��@)��֦��f?1�<��BV�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 18.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�58.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��n�?Il�J�lS@Q/�s�+�4@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�=Ab��?�=Ab��?!�=Ab��?      ��!       "	����( �?����( �?!����( �?*      ��!       2	�����?�����?!�����?:	��6p
@��6p
@!��6p
@B      ��!       J	f��@�9�?f��@�9�?!f��@�9�?R      ��!       Z	f��@�9�?f��@�9�?!f��@�9�?b      ��!       JGPUY��n�?b ql�J�lS@y/�s�+�4@