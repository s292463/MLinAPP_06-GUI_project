	9DܜZ@9DܜZ@!9DܜZ@	$�/�<^�?$�/�<^�?!$�/�<^�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL9DܜZ@$EdXś�?1+�m��@Ah�$���?I�"1A@Yı.n��?rEagerKernelExecute 0*	H�z��d@2F
Iterator::Model�K6l��?!FK�Z�=G@)t��%�?1��!.A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat C�*�?!�F5���=@)U�����?1<O�-E�9@:Preprocessing2U
Iterator::Model::ParallelMapV2�Gp#e��?!�\�-?(@)�Gp#e��?1�\�-?(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�[�O��?!d`��7&@)�[�O��?1d`��7&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�n��o�?!u1?�^1@)ݱ�&��?1�YMo@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Z�[!��?!��Y��J@)��	���?1���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorݵ�|гy?!Q�wt�U@)ݵ�|гy?1Q�wt�U@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZ*oG8-�?!��~k3@)X�|[�Tg?1�w����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 19.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�38.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9$�/�<^�?I\i7��bM@Q�׀K�C@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	$EdXś�?$EdXś�?!$EdXś�?      ��!       "	+�m��@+�m��@!+�m��@*      ��!       2	h�$���?h�$���?!h�$���?:	�"1A@�"1A@!�"1A@B      ��!       J	ı.n��?ı.n��?!ı.n��?R      ��!       Z	ı.n��?ı.n��?!ı.n��?b      ��!       JGPUY$�/�<^�?b q\i7��bM@y�׀K�C@