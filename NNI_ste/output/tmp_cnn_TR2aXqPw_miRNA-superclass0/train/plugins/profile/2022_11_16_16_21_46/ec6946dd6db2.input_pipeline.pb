	Z�'��@Z�'��@!Z�'��@	���n,@���n,@!���n,@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLZ�'��@������?1�r���v@A��[z�?I/��Ҙ@Yv4����?rEagerKernelExecute 0*	䥛� $g@2F
Iterator::Modelk�K�ƴ?!����W�E@)�)ʥ��?1�'��`@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�2����?!�g���8@@)�l���?1��cz�`<@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���ם�?!#6�L@)��n��?1v�	N�&@:Preprocessing2U
Iterator::Model::ParallelMapV2�ڧ�1�?!Z{g+&@)�ڧ�1�?1Z{g+&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�O��@�?!c+	qIz@)�O��@�?1c+	qIz@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�i�����?!x�7:��%@)��gB�Ă?1�}f��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�/�^|�~?!vGkԻA@)�/�^|�~?1vGkԻA@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��� 4J�?!��h+�(@)���;f?190o�t�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 17.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�38.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���n,@I�"�^�K@Q���2 �C@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	������?������?!������?      ��!       "	�r���v@�r���v@!�r���v@*      ��!       2	��[z�?��[z�?!��[z�?:	/��Ҙ@/��Ҙ@!/��Ҙ@B      ��!       J	v4����?v4����?!v4����?R      ��!       Z	v4����?v4����?!v4����?b      ��!       JGPUY���n,@b q�"�^�K@y���2 �C@