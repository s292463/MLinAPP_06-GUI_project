	�6��@�6��@!�6��@	vѷ!�?vѷ!�?!vѷ!�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�6��@?U�b��?1̘�5�f@A!XU/�Ӕ?I����ٱ@Y)x
�R��?rEagerKernelExecute 0*	�(\���c@2F
Iterator::Model�]0�掲?!;ƛ�P�F@)e6�$#g�?1�`�;K+?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�ީ�{��?!�N�v��<@)��߼8�?1EkoH�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatT:X��0�?!����"3@)�(yu��?1!����.@:Preprocessing2U
Iterator::Model::ParallelMapV2�	1�Tm�?!;W�V��,@)�	1�Tm�?1;W�V��,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���1�?!�9dL�:K@)�2��(}?1�so��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�E|'f�x?!�j�[@)�E|'f�x?1�j�[@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapʥ��$�?!;�.���>@)�s|�8ch?1��H#a��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor8fٓ��\?!�]�W��?)8fٓ��\?1�]�W��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceGsd��X?!�sBB,`�?)Gsd��X?1�sBB,`�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 16.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�41.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9vѷ!�?I�c�9�@M@Q��S���C@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	?U�b��??U�b��?!?U�b��?      ��!       "	̘�5�f@̘�5�f@!̘�5�f@*      ��!       2	!XU/�Ӕ?!XU/�Ӕ?!!XU/�Ӕ?:	����ٱ@����ٱ@!����ٱ@B      ��!       J	)x
�R��?)x
�R��?!)x
�R��?R      ��!       Z	)x
�R��?)x
�R��?!)x
�R��?b      ��!       JGPUYvѷ!�?b q�c�9�@M@y��S���C@