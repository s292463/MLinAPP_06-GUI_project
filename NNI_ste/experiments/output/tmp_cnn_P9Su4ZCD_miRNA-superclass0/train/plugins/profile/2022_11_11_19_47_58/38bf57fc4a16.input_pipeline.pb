	�U���n @�U���n @!�U���n @	�_ةc@�_ةc@!�_ةc@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�U���n @�({K9��?1`�+��&�?A����9�?IG�,�@Y���h o�?rEagerKernelExecute 0*	����kb@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�1 Ǟ�?!���?t�<@)%!���'�?1�� �8@:Preprocessing2U
Iterator::Model::ParallelMapV2�8+�&�?!��h�\8@)�8+�&�?1��h�\8@:Preprocessing2F
Iterator::Model�I/��?!5�+בF@)T �g�П?1U��zQ5@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�0_^�}�?!���(+@)�0_^�}�?1���(+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateUka�9�?!�1\�]3@)�tp�x�?1�~l��'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziph���c��?!��u�(nK@)d����~?1�1�M�z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǄ�K��{?!t>j�j]@)Ǆ�K��{?1t>j�j]@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����ҟ?!ٻ��/5@)��.�d?1}a���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 24.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�60.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�_ةc@I��$`�:U@Q������(@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�({K9��?�({K9��?!�({K9��?      ��!       "	`�+��&�?`�+��&�?!`�+��&�?*      ��!       2	����9�?����9�?!����9�?:	G�,�@G�,�@!G�,�@B      ��!       J	���h o�?���h o�?!���h o�?R      ��!       Z	���h o�?���h o�?!���h o�?b      ��!       JGPUY�_ةc@b q��$`�:U@y������(@