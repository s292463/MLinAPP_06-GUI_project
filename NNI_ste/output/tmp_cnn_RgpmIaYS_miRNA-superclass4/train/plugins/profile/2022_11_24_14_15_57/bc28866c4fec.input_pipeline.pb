	��V��!@��V��!@!��V��!@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC��V��!@~5�h�?1��o�@A��G�Ȱ�?I�S���@rEagerKernelExecute 0*	�K7�ATb@2F
Iterator::ModelK�|%��?!�)�N|gI@)D�1uWv�?1���(�@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�x]�`7�?!��1x��:@)C�K��?1T1��G6@:Preprocessing2U
Iterator::Model::ParallelMapV2�����]�?!�O/}��0@)�����]�?1�O/}��0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���<�?!!��[�]!@)���<�?1!��[�]!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����E�?!	/+�Ѫ-@)@KW��x�?1�۰�X�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipT^-w�?!6����H@)��Y�e�?1"$���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����{?!c�3�@)����{?1c�3�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapj�L�:�?!�-��0@)���1��g?1@d��?��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 12.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�41.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI^�~�2�K@Q�^�O�&F@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	~5�h�?~5�h�?!~5�h�?      ��!       "	��o�@��o�@!��o�@*      ��!       2	��G�Ȱ�?��G�Ȱ�?!��G�Ȱ�?:	�S���@�S���@!�S���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q^�~�2�K@y�^�O�&F@