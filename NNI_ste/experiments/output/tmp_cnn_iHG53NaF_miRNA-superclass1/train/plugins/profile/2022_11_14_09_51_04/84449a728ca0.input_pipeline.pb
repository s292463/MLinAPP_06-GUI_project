	"��Լ@"��Լ@!"��Լ@	�Њ@�Њ@!�Њ@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL"��Լ@����Bt�?1��PN��@A���BΛ?I	4��@Y�Ƥ���?rEagerKernelExecute 0*	Q��nWb@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���Hi�?!Sc��_�=@)�x\T���?1��K۫8@:Preprocessing2F
Iterator::Model`s�	M�?!1&��s�A@)����`��?1,S-�zh4@:Preprocessing2U
Iterator::Model::ParallelMapV2�"���?!l��4-@)�"���?1l��4-@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice`;�O �?!�9����+@)`;�O �?1�9����+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy���h�?!��F?P@)mu9% &�?1��|�&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�T���B�?!�����5@)��`��
�?1;�U�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor#h�$�?!T�"�@)#h�$�?1T�"�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�����ա?!�+���7@)�P�,i?1L�ȸo� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�41.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s7.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�Њ@I�+{.l�H@Q(�9�F@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����Bt�?����Bt�?!����Bt�?      ��!       "	��PN��@��PN��@!��PN��@*      ��!       2	���BΛ?���BΛ?!���BΛ?:		4��@	4��@!	4��@B      ��!       J	�Ƥ���?�Ƥ���?!�Ƥ���?R      ��!       Z	�Ƥ���?�Ƥ���?!�Ƥ���?b      ��!       JGPUY�Њ@b q�+{.l�H@y(�9�F@