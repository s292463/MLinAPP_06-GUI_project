	(5
�@(5
�@!(5
�@	eu5@eu5@!eu5@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL(5
�@�`7l[��?1+�&�|��?A�T�z��?I��� �6@YcAJ��?rEagerKernelExecute 0*	���S�c@2F
Iterator::Model��o�ㆳ?!/���2H@)QlMK��?1���5E�A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�b����?!$����@@)��\k�?1�R�$D�;@:Preprocessing2U
Iterator::Model::ParallelMapV24��7�?!�����)@)4��7�?1�����)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice~�֤��?!����@)~�֤��?1����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�#H��Ѵ?!�*�(�I@)��2�?1�{�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���N���?!�ua(Qm)@)�-���?1T��H��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�*n|?!���Ɲ@)�J�*n|?1���Ɲ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���W�<�?!�T�L��,@)U�wE�e?1h��"!��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 25.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�45.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9fu5@I�zO˽Q@Qӆ����8@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�`7l[��?�`7l[��?!�`7l[��?      ��!       "	+�&�|��?+�&�|��?!+�&�|��?*      ��!       2	�T�z��?�T�z��?!�T�z��?:	��� �6@��� �6@!��� �6@B      ��!       J	cAJ��?cAJ��?!cAJ��?R      ��!       Z	cAJ��?cAJ��?!cAJ��?b      ��!       JGPUYfu5@b q�zO˽Q@yӆ����8@