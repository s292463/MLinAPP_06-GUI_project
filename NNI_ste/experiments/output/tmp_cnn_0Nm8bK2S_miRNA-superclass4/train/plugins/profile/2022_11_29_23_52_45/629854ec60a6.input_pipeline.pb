	a�N"�"@a�N"�"@!a�N"�"@	�U����#@�U����#@!�U����#@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCa�N"�"@���R$_�?1��|ԋ@Ij��{@Yw��׹i�?rEagerKernelExecute 0*	l����d@2F
Iterator::Model�q75�?!q�J�P�G@)c|��l;�?1r0=&�@A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat4�IbI��?!9�\@@)���=��?1�q�hB<@:Preprocessing2U
Iterator::Model::ParallelMapV2p��^�?!�7��f*@)p��^�?1�7��f*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�ƻ#c��?!ȡ�{X@)�ƻ#c��?1ȡ�{X@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�)�ޕ?!�P�p�)@)섗���?1 p(eH@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipu��&�?!�
�e�%J@)�R\U�]�?1 ����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor9% &�B~?!O�+��@)9% &�B~?1O�+��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapfh<�y�?!X�(��,@)��x@�d?1Ys�_��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 9.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�33.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�U����#@I�~Q�mB@Q��ufK@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���R$_�?���R$_�?!���R$_�?      ��!       "	��|ԋ@��|ԋ@!��|ԋ@*      ��!       2      ��!       :	j��{@j��{@!j��{@B      ��!       J	w��׹i�?w��׹i�?!w��׹i�?R      ��!       Z	w��׹i�?w��׹i�?!w��׹i�?b      ��!       JGPUY�U����#@b q�~Q�mB@y��ufK@