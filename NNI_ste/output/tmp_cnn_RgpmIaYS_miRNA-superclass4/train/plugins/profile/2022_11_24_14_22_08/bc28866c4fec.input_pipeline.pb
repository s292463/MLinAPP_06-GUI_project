	������*@������*@!������*@	���b��?���b��?!���b��?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL������*@?�a�'�?1���X$@A����t�?I$d �.��?Y���Fu:�?rEagerKernelExecute 0*	㥛� �c@2F
Iterator::Model4h��b�?!��c,�TJ@):tzލ�?1�x���A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����,A�?!_=���f;@)ۧ�1��?17[�5@:Preprocessing2U
Iterator::Model::ParallelMapV2Z���f��?!Q��YB�0@)Z���f��?1Q��YB�0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice^���?!��F��@)^���?1��F��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��X����?!����@)��X����?1����@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateI�\߇��?!���A)@)~�[�~l�?1N��P3�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipsJ_9�?!(+��	�G@)r�Z|
��?1�����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapB]¡�?!Z0k�-@)R~R���h?1�c򉒶�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 9.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�12.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���b��?Iě=���6@Q��$���R@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	?�a�'�??�a�'�?!?�a�'�?      ��!       "	���X$@���X$@!���X$@*      ��!       2	����t�?����t�?!����t�?:	$d �.��?$d �.��?!$d �.��?B      ��!       J	���Fu:�?���Fu:�?!���Fu:�?R      ��!       Z	���Fu:�?���Fu:�?!���Fu:�?b      ��!       JGPUY���b��?b qě=���6@y��$���R@