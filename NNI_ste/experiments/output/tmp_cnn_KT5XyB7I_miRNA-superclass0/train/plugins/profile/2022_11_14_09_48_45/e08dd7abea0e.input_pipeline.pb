	cԵ�>5 @cԵ�>5 @!cԵ�>5 @      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCcԵ�>5 @G��R^��?1E��f��?Aj�����?I}ԛQ#@rEagerKernelExecute 0*	��"���c@2U
Iterator::Model::ParallelMapV2Ͼ� =E�?!�����;@)Ͼ� =E�?1�����;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat4�����?!���g	8;@)�uoEb��?1�1@ ��6@:Preprocessing2F
Iterator::Model��!���?!�P��uGG@)�L�^�i�?1���@�2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateOX�eS�?!P^/?�2@)�`���?1B��\H�%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�lw�N�?!�rlp@)�lw�N�?1�rlp@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipQ�5�U��?!1�6��J@)�F�@�?1��un@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�{L�4{?!�ڝ�@)�{L�4{?1�ڝ�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap� [��ˠ?!b+����4@)I�V�j?1��Hy&: @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 22.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�62.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noII6��RU@Q�M�
�k-@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	G��R^��?G��R^��?!G��R^��?      ��!       "	E��f��?E��f��?!E��f��?*      ��!       2	j�����?j�����?!j�����?:	}ԛQ#@}ԛQ#@!}ԛQ#@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qI6��RU@y�M�
�k-@