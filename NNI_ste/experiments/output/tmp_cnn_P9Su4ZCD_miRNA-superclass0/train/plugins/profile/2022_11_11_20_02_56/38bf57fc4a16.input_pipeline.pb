	��ky�Z@��ky�Z@!��ky�Z@	���;�@���;�@!���;�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��ky�Z@���G��?1��6��@A�����?I���+	@Y��[[�?rEagerKernelExecute 0*	��x�&�q@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenaterP�Lۿ�?!�5��NL@)��g�e�?1/
ɆI@:Preprocessing2F
Iterator::ModelK��F>��?!�*2�6@)s���M�?1n�*���/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�׼��Z�?!Br��0@)���jׄ�?1�2YV`�+@:Preprocessing2U
Iterator::Model::ParallelMapV2%�I(}!�?!���v@)%�I(}!�?1���v@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���(_В?!$[)�@)���(_В?1$[)�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���N�?!~|u3OS@)M0�k���?1�{_�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorc�: �~?!u�+�&�@)c�: �~?1u�+�&�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap,�`p��?!�?��L@)m���|g?1�TRh�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 20.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�44.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��;�@I���d�CP@Q�ފ
�{?@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���G��?���G��?!���G��?      ��!       "	��6��@��6��@!��6��@*      ��!       2	�����?�����?!�����?:	���+	@���+	@!���+	@B      ��!       J	��[[�?��[[�?!��[[�?R      ��!       Z	��[[�?��[[�?!��[[�?b      ��!       JGPUY��;�@b q���d�CP@y�ފ
�{?@