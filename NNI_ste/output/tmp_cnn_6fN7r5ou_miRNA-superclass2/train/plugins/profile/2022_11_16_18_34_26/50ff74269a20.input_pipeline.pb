	x�����!@x�����!@!x�����!@	�Q����@�Q����@!�Q����@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLx�����!@��u��?1�fG��@A��[�O�?IEf.pyL@Y�d�F ^�?rEagerKernelExecute 0*	;�O��nd@2F
Iterator::Model&m��ͱ?!#�}|�EE@)�%Tp�?1 �;c�3=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�7�-:Y�?!	����{?@)͓k
dv�?1`����:@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea�����?!��
gVU+@)a�����?1��
gVU+@:Preprocessing2U
Iterator::Model::ParallelMapV2y��n�U�?!M��+�*@)y��n�U�?1M��+�*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�S���
�?!���0�L@)�=~o�?1�(D/9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatev��^
�?!ͪ���2@)+�)�T�?1Ak]Fu�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorm�?!����@)m�?1����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�?�C�?!xriZA*4@)&:�,B�e?1�z�T���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 14.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�38.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�Q����@I��|H{J@Q"a1F@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��u��?��u��?!��u��?      ��!       "	�fG��@�fG��@!�fG��@*      ��!       2	��[�O�?��[�O�?!��[�O�?:	Ef.pyL@Ef.pyL@!Ef.pyL@B      ��!       J	�d�F ^�?�d�F ^�?!�d�F ^�?R      ��!       Z	�d�F ^�?�d�F ^�?!�d�F ^�?b      ��!       JGPUY�Q����@b q��|H{J@y"a1F@