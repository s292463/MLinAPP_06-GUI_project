	�n�EE!@�n�EE!@!�n�EE!@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�n�EE!@S�[�z@1� U�?A)�A&9�?I�A	3m@rEagerKernelExecute 0*	d;�O�`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatdWZF�=�?!Kk�!)@@)rm��o�?1�5�Ǿ<@:Preprocessing2F
Iterator::Model�HP��?!�Ə��D@)�vۅ�:�?1¯���<6@:Preprocessing2U
Iterator::Model::ParallelMapV2,����?!X��mI�2@),����?1X��mI�2@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice$�@��?!�`�Z�"@)$�@��?1�`�Z�"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��ĭ��?!�v5��T2@)�o����?1���g��!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���O�?!�69pbbM@)�an�r?1k�@ �@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�O�mpv?!ぱ4@)�O�mpv?1ぱ4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��vۅ�?!Ҧ�G9w4@)�O�mpf?1ぱ4@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 25.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�56.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI����(�T@Q~yA�\�1@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	S�[�z@S�[�z@!S�[�z@      ��!       "	� U�?� U�?!� U�?*      ��!       2	)�A&9�?)�A&9�?!)�A&9�?:	�A	3m@�A	3m@!�A	3m@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q����(�T@y~yA�\�1@