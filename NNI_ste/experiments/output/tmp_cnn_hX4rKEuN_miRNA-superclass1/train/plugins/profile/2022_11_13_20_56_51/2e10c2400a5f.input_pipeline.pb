	s��h��@s��h��@!s��h��@	��[�*
@��[�*
@!��[�*
@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLs��h��@�1^��?1��vKr �?A��A$C��?I�!6X8I@Y-|}�K��?rEagerKernelExecute 0*	���S/f@2F
Iterator::Model���6�ٳ?!Ȓ����E@)A(��h��?1���l?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�V&�R?�?!�ҳS�9@)P��|zl�?1�`5@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�*�MF��?!8mb\'L@)2Ƈ�˖?1�P6�?)@:Preprocessing2U
Iterator::Model::ParallelMapV2�>#K�?!�lt��(@)�>#K�?1�lt��(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceY�|^�?!���'@)Y�|^�?1���'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���;��?!�����N0@)q�{��c�?1���#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorݲC�Ö~?!/$�:��@)ݲC�Ö~?1/$�:��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapy�z�L��?!�#$E.2@)GW��:k?1�B��c��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 15.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�59.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��[�*
@I�y���S@Q-]��_4@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�1^��?�1^��?!�1^��?      ��!       "	��vKr �?��vKr �?!��vKr �?*      ��!       2	��A$C��?��A$C��?!��A$C��?:	�!6X8I@�!6X8I@!�!6X8I@B      ��!       J	-|}�K��?-|}�K��?!-|}�K��?R      ��!       Z	-|}�K��?-|}�K��?!-|}�K��?b      ��!       JGPUY��[�*
@b q�y���S@y-]��_4@