	�wD @�wD @!�wD @	'k\���?'k\���?!'k\���?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�wD @DkE����?1���>9��?A b���4�?I�/����@Y���z0�?rEagerKernelExecute 0*	��C�l�b@2F
Iterator::Model�;k�]h�?!����!H@)/��d�۪?13b� �A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatp&����?!�
g�B1:@)H1@�	�?1�����c6@:Preprocessing2U
Iterator::Model::ParallelMapV2�p��?!�b5D*@)�p��?1�b5D*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip<FzQ��?!Y.�I@)�� 4J��?1�P��U$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4�?!�=����@)�J�4�?1�=����@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�j���?!�C`���+@)���y7�?1�I(�XU@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;�/K;5w?!�\x��l@);�/K;5w?1�\x��l@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapȳ˷>��?!Xʙ��/@)j0�G�d?1�5�9�9�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�55.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9'k\���?I��]��S@Q���~�3@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	DkE����?DkE����?!DkE����?      ��!       "	���>9��?���>9��?!���>9��?*      ��!       2	 b���4�? b���4�?! b���4�?:	�/����@�/����@!�/����@B      ��!       J	���z0�?���z0�?!���z0�?R      ��!       Z	���z0�?���z0�?!���z0�?b      ��!       JGPUY'k\���?b q��]��S@y���~�3@