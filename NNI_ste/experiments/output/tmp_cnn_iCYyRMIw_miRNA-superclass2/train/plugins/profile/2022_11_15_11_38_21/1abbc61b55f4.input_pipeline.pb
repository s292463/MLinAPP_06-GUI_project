	�yUg�@�yUg�@!�yUg�@	��ϋh�@��ϋh�@!��ϋh�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�yUg�@PQ�+��?1�	�?@A~;��"�?I�EE�N�@Ye����?rEagerKernelExecute 0*	�n���r@2U
Iterator::Model::ParallelMapV2+����?!ٱ�#C�L@)+����?1ٱ�#C�L@:Preprocessing2F
Iterator::Model�)��?!QV�kmR@)�?�:s�?1��A�(s0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��g���?!��X��+@)�I�?���?1�uf�?'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�g���?!���QJ:@)���%�?1Xp
��(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�$�j�?!����@)�$�j�?1����@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate\�3�?O�?!����Y@)�`���?1��WK3@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor\��AA)z?!�J�C#,@)\��AA)z?1�J�C#,@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��%���?!]_ћ�@)ګ����e?1
Ns�.��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�37.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t19.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9��ϋh�@Iw���L@QS�"�B@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	PQ�+��?PQ�+��?!PQ�+��?      ��!       "	�	�?@�	�?@!�	�?@*      ��!       2	~;��"�?~;��"�?!~;��"�?:	�EE�N�@�EE�N�@!�EE�N�@B      ��!       J	e����?e����?!e����?R      ��!       Z	e����?e����?!e����?b      ��!       JGPUY��ϋh�@b qw���L@yS�"�B@