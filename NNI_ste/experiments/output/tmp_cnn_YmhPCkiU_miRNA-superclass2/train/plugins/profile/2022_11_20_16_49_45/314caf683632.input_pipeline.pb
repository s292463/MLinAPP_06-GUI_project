	��L��@��L��@!��L��@	�!�6@�!�6@!�!�6@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��L��@��oBa�?1`�E���?A�����?I�h�^`�@Yж�u���?rEagerKernelExecute 0*	�l���5c@2F
Iterator::Model����K��?!�Hm1�hI@)9|҉S�?1�Xj q�B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate#/kb���?!����_?@)�s)�*��?1�h*�5=@:Preprocessing2U
Iterator::Model::ParallelMapV2���&S�?!X�D0+@)���&S�?1X�D0+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatsd���?!�1y��%@)F�7�k�?1�TJ,9@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�LnY�?!i����H@)�d���}?1���a��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�g���y?!�I��;@)�g���y?1�I��;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��>rk�?!�����@@)��h��k?1#w����@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor ����]?!._��?) ����]?1._��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice~Q��B�X?!U��2�6�?)~Q��B�X?1U��2�6�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 24.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�46.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�!�6@I������Q@Q��9g8@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��oBa�?��oBa�?!��oBa�?      ��!       "	`�E���?`�E���?!`�E���?*      ��!       2	�����?�����?!�����?:	�h�^`�@�h�^`�@!�h�^`�@B      ��!       J	ж�u���?ж�u���?!ж�u���?R      ��!       Z	ж�u���?ж�u���?!ж�u���?b      ��!       JGPUY�!�6@b q������Q@y��9g8@