	�gx�� @�gx�� @!�gx�� @	��}"l@��}"l@!��}"l@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�gx�� @Քd�.�?1]�`7l�@A���5x?I���G� @Y�;�y�9�?rEagerKernelExecute 0*	�E���Ձ@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�}q�J[�?!U9EJ�R@)�//�>:�?1�م<ݔP@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatvQ���`�?!�#�G$@)=֌r�?1T��dA,!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�rJ_�?!��m�!!@)�rJ_�?1��m�!!@:Preprocessing2F
Iterator::Model/�o��e�?!�ZZw�'@) �^EF�?1�t����@:Preprocessing2U
Iterator::Model::ParallelMapV2��`��
�?!�j�Q�@)��`��
�?1�j�Q�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�;����?!\���V@))"�*�Ȍ?1>u-LO�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���V%�?!V�h�1x�?)���V%�?1V�h�1x�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA�} R��?!��.���R@)���p?1� ��-��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 6.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�25.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��}"l@I�����b?@Qʬ�*�3P@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Քd�.�?Քd�.�?!Քd�.�?      ��!       "	]�`7l�@]�`7l�@!]�`7l�@*      ��!       2	���5x?���5x?!���5x?:	���G� @���G� @!���G� @B      ��!       J	�;�y�9�?�;�y�9�?!�;�y�9�?R      ��!       Z	�;�y�9�?�;�y�9�?!�;�y�9�?b      ��!       JGPUY��}"l@b q�����b?@yʬ�*�3P@