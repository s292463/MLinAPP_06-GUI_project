	�IF��6#@�IF��6#@!�IF��6#@	X���>@X���>@!X���>@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�IF��6#@w� ݗ3�?1��<�9@AbJ$��(v?I�$[]N�@Y�s�v���?rEagerKernelExecute 0*	��Q��u@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�h�wa�?!a�i���P@)�8�Z�?1�0�A�P@:Preprocessing2F
Iterator::Model@�� kղ?!��WI�S5@)p|�%�?1��u�r-@:Preprocessing2U
Iterator::Model::ParallelMapV2"p$�`S�?!ʰ^:&j@)"p$�`S�?1ʰ^:&j@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����)��?!k{�b%@)����)��?1k{�b%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��:q9^�?!�-�S@)��%�`�?15,.	�W	@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�|?q �?!�%q��k @)j�q���?1}�{e@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorR<��k�?!��y�m�@)R<��k�?1��y�m�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap1?74e��?!��_@�!@)+~��7e?1��>���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�24.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9W���>@I0��;6L;@Q�|��iP@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	w� ݗ3�?w� ݗ3�?!w� ݗ3�?      ��!       "	��<�9@��<�9@!��<�9@*      ��!       2	bJ$��(v?bJ$��(v?!bJ$��(v?:	�$[]N�@�$[]N�@!�$[]N�@B      ��!       J	�s�v���?�s�v���?!�s�v���?R      ��!       Z	�s�v���?�s�v���?!�s�v���?b      ��!       JGPUYW���>@b q0��;6L;@y�|��iP@