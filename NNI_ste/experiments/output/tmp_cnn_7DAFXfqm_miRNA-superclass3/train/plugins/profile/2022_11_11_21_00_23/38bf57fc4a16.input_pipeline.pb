	m�s��@m�s��@!m�s��@	�gyQ@�gyQ@!�gyQ@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLm�s��@��8�j��?1�xy:��?A��>s֧�?I}�;l"�@YT� �!��?rEagerKernelExecute 0*	{�G�ra@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatȗP��?!f���>@)/ܹ0ҋ�?1�@���9@:Preprocessing2F
Iterator::Model^��v1�?!)?�blD@)e�fb�?1:��"�9@:Preprocessing2U
Iterator::Model::ParallelMapV2��W:��?!0��`E?.@)��W:��?10��`E?.@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���y7�?!5�
�,@)���y7�?15�
�,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateN�����?!e�g3�4@)/�
ҌE�?1�k��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipJ��Gp#�?!���/��M@)�ݮ���?1H�o,o@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�^��x�z?!^39�@�@)�^��x�z?1^39�@�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��b�?!)�Z>��6@)j�drjgh?1�葴�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 25.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�46.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�gyQ@IS��l�R@Q��d,;9@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��8�j��?��8�j��?!��8�j��?      ��!       "	�xy:��?�xy:��?!�xy:��?*      ��!       2	��>s֧�?��>s֧�?!��>s֧�?:	}�;l"�@}�;l"�@!}�;l"�@B      ��!       J	T� �!��?T� �!��?!T� �!��?R      ��!       Z	T� �!��?T� �!��?!T� �!��?b      ��!       JGPUY�gyQ@b qS��l�R@y��d,;9@