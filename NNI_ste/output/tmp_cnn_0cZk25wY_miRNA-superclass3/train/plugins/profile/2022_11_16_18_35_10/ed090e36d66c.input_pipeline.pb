	V-�(#@V-�(#@!V-�(#@	���2��@���2��@!���2��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLV-�(#@T����c�?1d�1>@AvS�k%t�?I�XP��@Y�5|��?rEagerKernelExecute 0*	z�&1�g@2F
Iterator::ModelE�a���?!^l��E@)y�'eR�?1�L盧�;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���GS=�?!���_��9@)��x���?1��%|:6@:Preprocessing2U
Iterator::Model::ParallelMapV2$*T7�?!���Ik',@)$*T7�?1���Ik',@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate@��߼�?!r��PT9@)��b.�?1f����)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceA�c�]K�?!}�˩�(@)A�c�]K�?1}�˩�(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�YL�?!��_Q�L@)MHk:!�?1�{�1u�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�?x�=|?!0;����@)�?x�=|?10;����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/O�R�?!h����:@)_��x�Zi?1d
N���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 14.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�30.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���2��@IZ��Y�F@Q.ã�NI@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	T����c�?T����c�?!T����c�?      ��!       "	d�1>@d�1>@!d�1>@*      ��!       2	vS�k%t�?vS�k%t�?!vS�k%t�?:	�XP��@�XP��@!�XP��@B      ��!       J	�5|��?�5|��?!�5|��?R      ��!       Z	�5|��?�5|��?!�5|��?b      ��!       JGPUY���2��@b qZ��Y�F@y.ã�NI@