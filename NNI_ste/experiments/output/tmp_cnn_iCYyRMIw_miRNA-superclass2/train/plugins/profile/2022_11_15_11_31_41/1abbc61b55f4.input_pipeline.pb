	�!�� �@�!�� �@!�!�� �@	�0��C@�0��C@!�0��C@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�!�� �@���W��?1���`�#@A��I�?��?Isd���@Y?q ���?rEagerKernelExecute 0*	/�$�d@2F
Iterator::ModelO�}���?!�����I@)�e2�g�?1���C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��f�|�?!R��B֐9@)Œr�9>�?14H	���5@:Preprocessing2U
Iterator::Model::ParallelMapV2�����?!],�r�(@)�����?1],�r�(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��G�?!>�)�
�%@)��G�?1>�)�
�%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��H��_�?!(�kW~=H@)�w��!�?1P4.�y�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�����?!�i�:��.@)̛õ��~?1[ϷNOM@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�v|��y?!�X��@)�v|��y?1�X��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZ�N��?!�H91@)@�j��g?1Boϝ�f�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 19.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�41.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�0��C@I��~�G�N@QP�-�OrA@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���W��?���W��?!���W��?      ��!       "	���`�#@���`�#@!���`�#@*      ��!       2	��I�?��?��I�?��?!��I�?��?:	sd���@sd���@!sd���@B      ��!       J	?q ���??q ���?!?q ���?R      ��!       Z	?q ���??q ���?!?q ���?b      ��!       JGPUY�0��C@b q��~�G�N@yP�-�OrA@