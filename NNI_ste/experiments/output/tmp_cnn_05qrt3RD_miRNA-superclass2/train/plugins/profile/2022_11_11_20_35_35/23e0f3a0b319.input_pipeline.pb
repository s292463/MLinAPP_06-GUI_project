	9�#+?;d@9�#+?;d@!9�#+?;d@	c�!o��?c�!o��?!c�!o��?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL9�#+?;d@�뤾,-�?1q���hk`@AEH�ξ�?I�r���<@Y ��Ud�?rEagerKernelExecute 0*	Zd;�O�_@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�B:<��?!��t�XE@)�hE,�?1�BN��C@:Preprocessing2F
Iterator::Model���R�?!u���(
C@)a�.�e��?1�
)"�6@:Preprocessing2U
Iterator::Model::ParallelMapV2���g?R�?!��p�^/@)���g?R�?1��p�^/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatR&5�ؐ?!�ϫ2�)@)�[��.��?1uۇ �� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��P��C�?!�K;��N@){��x?1����l@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���iw?!��G��@)���iw?1��G��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���?�?!o 99F@)����[b?1�1����?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor�A�p�-^?!�����?)�A�p�-^?1�����?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice����yZ?!����9�?)����yZ?1����9�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�17.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9c�!o��?I�WL;r�2@Q���9(JT@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�뤾,-�?�뤾,-�?!�뤾,-�?      ��!       "	q���hk`@q���hk`@!q���hk`@*      ��!       2	EH�ξ�?EH�ξ�?!EH�ξ�?:	�r���<@�r���<@!�r���<@B      ��!       J	 ��Ud�? ��Ud�?! ��Ud�?R      ��!       Z	 ��Ud�? ��Ud�?! ��Ud�?b      ��!       JGPUYc�!o��?b q�WL;r�2@y���9(JT@