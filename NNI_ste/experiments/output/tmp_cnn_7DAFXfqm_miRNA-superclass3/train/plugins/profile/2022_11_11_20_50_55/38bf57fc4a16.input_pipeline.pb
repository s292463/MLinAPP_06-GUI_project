	��yw@��yw@!��yw@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC��yw@ <�Bu��?1<����?A��6��?Irjg���@rEagerKernelExecute 0*	n���Wr@2Z
#Iterator::Model::ParallelMapV2::Zip{�p̲'�?!·�;��T@)~9�]��?1d�/q�\M@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�;k�]h�?!l�l��-@)����բ?1DEND�)@:Preprocessing2F
Iterator::Model��c"��?!��1@)�d��~��?1�6, f�#@:Preprocessing2U
Iterator::Model::ParallelMapV2q��Ŧ�?!��?P�@)q��Ŧ�?1��?P�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���9̗?!voU���@)���:�f�?1���==@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceo�ŏ1�?!� ����@)o�ŏ1�?1� ����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorR臭��|?!���\@)R臭��|?1���\@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapx���?!s�T�t�"@)���X�p?1�R�%	�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 21.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�50.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI|Im�KR@Q�J���;@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	 <�Bu��? <�Bu��?! <�Bu��?      ��!       "	<����?<����?!<����?*      ��!       2	��6��?��6��?!��6��?:	rjg���@rjg���@!rjg���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q|Im�KR@y�J���;@