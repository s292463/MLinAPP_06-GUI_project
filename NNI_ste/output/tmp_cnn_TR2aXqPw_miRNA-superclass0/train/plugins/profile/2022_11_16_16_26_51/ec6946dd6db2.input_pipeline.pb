	�pu ��@�pu ��@!�pu ��@	��5�@��5�@!��5�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�pu ��@S=��M�?13�ۃ� @A$)�ahu�?IPP�V�%@Y������?rEagerKernelExecute 0*	���Qdr@2U
Iterator::Model::ParallelMapV2j�WV���?!����qF@)j�WV���?1����qF@:Preprocessing2F
Iterator::Model�4���?!�PTL�O@)����0��?1�Ə��3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatzo���?!�+5`?	5@)�ص�ݒ�?1H[�o�2@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceg,��N�?!H��S�@)g,��N�?1H��S�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�3�ۃ�?!%�W�7�%@)���k��?1�j_8�
@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�;FzQ�?!�;���!B@)7��:r��?1�-��
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor,���x?!���ߑ @),���x?1���ߑ @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�v��?!�O�J��'@)�/��Ch?1��}1��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 20.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�44.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��5�@I�GkG]P@Q���=�?@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	S=��M�?S=��M�?!S=��M�?      ��!       "	3�ۃ� @3�ۃ� @!3�ۃ� @*      ��!       2	$)�ahu�?$)�ahu�?!$)�ahu�?:	PP�V�%@PP�V�%@!PP�V�%@B      ��!       J	������?������?!������?R      ��!       Z	������?������?!������?b      ��!       JGPUY��5�@b q�GkG]P@y���=�?@