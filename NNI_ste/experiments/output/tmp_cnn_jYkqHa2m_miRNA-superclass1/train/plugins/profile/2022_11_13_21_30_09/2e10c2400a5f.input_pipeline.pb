	G8-x�_"@G8-x�_"@!G8-x�_"@		���#b@	���#b@!	���#b@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCG8-x�_"@jM�S��?1����4�@I��7��X @Yh��n�?rEagerKernelExecute 0*	H�z��b@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�G��?!C�g���B@)]lZ)r�?1���[��@@:Preprocessing2F
Iterator::Modelv8�Jwש?!�ض���@@)l�u���?1{9^H;3@:Preprocessing2U
Iterator::Model::ParallelMapV2l��+�?!���4�,@)l��+�?1���4�,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceXr���?!/n,v$�,@)Xr���?1/n,v$�,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateÂ��?!ɓg��4@)[&��|�?1�r�� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip;�s��q�?!������P@)8�:V)=�?1�۾^�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorZ���аx?!�5�T�@)Z���аx?1�5�T�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��E;��?![c��6@)�]���h?1(��0�J�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 7.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�22.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���#b@I�� �>@Q�����P@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	jM�S��?jM�S��?!jM�S��?      ��!       "	����4�@����4�@!����4�@*      ��!       2      ��!       :	��7��X @��7��X @!��7��X @B      ��!       J	h��n�?h��n�?!h��n�?R      ��!       Z	h��n�?h��n�?!h��n�?b      ��!       JGPUY���#b@b q�� �>@y�����P@