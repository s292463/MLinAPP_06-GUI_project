	���� @���� @!���� @	~b�lw� @~b�lw� @!~b�lw� @"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL���� @�x�ߢ��?14�i���@AE�4~ᕔ?IlЗ��\�?Y�ڧ�1��?rEagerKernelExecute 0*	��|?5d@2F
Iterator::Model�_��9�?!����K@)�[w�T��?1���D@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateϢw*���?!�P`�<@)�����?1@��-e�:@:Preprocessing2U
Iterator::Model::ParallelMapV2bX9�Ȗ?!�r�K�+@)bX9�Ȗ?1�r�K�+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat ���7�?!��	�]-&@)4�����?1�3���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�	�Yٲ?!F�7�F@)�L�T�#{?1zD��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�`�y?!�WnQ@)�`�y?1�WnQ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�9$�P2�?!���]!�>@)�lɪ7i?1R>��߱�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor*6�u�![?!�ISڠ��?)*6�u�![?1�ISڠ��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�I+�V?!f���l�?)�I+�V?1f���l�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 8.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�23.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t13.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9}b�lw� @I'��>PqB@Q9]-��aK@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�x�ߢ��?�x�ߢ��?!�x�ߢ��?      ��!       "	4�i���@4�i���@!4�i���@*      ��!       2	E�4~ᕔ?E�4~ᕔ?!E�4~ᕔ?:	lЗ��\�?lЗ��\�?!lЗ��\�?B      ��!       J	�ڧ�1��?�ڧ�1��?!�ڧ�1��?R      ��!       Z	�ڧ�1��?�ڧ�1��?!�ڧ�1��?b      ��!       JGPUY}b�lw� @b q'��>PqB@y9]-��aK@