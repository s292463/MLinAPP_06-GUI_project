	'ݖ��"@'ݖ��"@!'ݖ��"@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC'ݖ��"@�`��
@1��-I)@AT:X��0�?It_�lW�@rEagerKernelExecute 0*	|?5^�}c@2F
Iterator::Model�.��?!�g���gJ@)E+��B�?1���?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatb���4�?!�``��9@)Z��U�P�?1.9��O�5@:Preprocessing2U
Iterator::Model::ParallelMapV2��Z��?!ʧ�+5@)��Z��?1ʧ�+5@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��Q�Q�?!z�Y3�4@)��Q�Q�?1z�Y3�4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�x$^�Ε?!_����P+@)~��7L�?1D+�l@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�b�T4ֲ?!q�U*�G@)z����?1�ӫ5U�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorC p��sy?!8�˔�@)C p��sy?18�˔�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����{�?!J���.@)����ce?1��E����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 35.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�41.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��"�_BS@Q'�u���6@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�`��
@�`��
@!�`��
@      ��!       "	��-I)@��-I)@!��-I)@*      ��!       2	T:X��0�?T:X��0�?!T:X��0�?:	t_�lW�@t_�lW�@!t_�lW�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��"�_BS@y'�u���6@