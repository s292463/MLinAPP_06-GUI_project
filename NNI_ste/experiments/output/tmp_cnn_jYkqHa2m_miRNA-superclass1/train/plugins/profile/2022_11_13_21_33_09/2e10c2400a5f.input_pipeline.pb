	��K�A�'@��K�A�'@!��K�A�'@	1�ݚ@1�ݚ@!1�ݚ@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��K�A�'@�g͏�4�?1wLݕ]�@AUka�9�?I4e���@Y�;jL���?rEagerKernelExecute 0*	��n��`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�/K;5��?!��ϲ�<@)�TގpZ�?1`�C��7@:Preprocessing2F
Iterator::ModelI.�!���?!9����E@)�.R(_�?1��!Ϝ�6@:Preprocessing2U
Iterator::Model::ParallelMapV2�-�?!���b�4@)�-�?1���b�4@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�ʉv�?!�O��1a*@)�ʉv�?1�O��1a*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatew����?!��i\4@)s��+܂?1����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipM�:�/K�?!�:�F %L@)U���)~?1��c��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��fc%�y?!�!0���@)��fc%�y?1�!0���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap̸���s�?!���66@)��v�ӂg?1=`ˀ�%@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 12.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�31.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no91�ݚ@IS�̯F@Q���Y��J@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�g͏�4�?�g͏�4�?!�g͏�4�?      ��!       "	wLݕ]�@wLݕ]�@!wLݕ]�@*      ��!       2	Uka�9�?Uka�9�?!Uka�9�?:	4e���@4e���@!4e���@B      ��!       J	�;jL���?�;jL���?!�;jL���?R      ��!       Z	�;jL���?�;jL���?!�;jL���?b      ��!       JGPUY1�ݚ@b qS�̯F@y���Y��J@