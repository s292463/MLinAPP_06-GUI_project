	P6�
?@P6�
?@!P6�
?@	T�@���@T�@���@!T�@���@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLP6�
?@f���8��?1�闈�� @AƧ Ϡ�?IN�f�}@YV�����?rEagerKernelExecute 0*	��C��^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���N�?![���ʔA@)4��X�_�?1�PK��N=@:Preprocessing2F
Iterator::Model��E
e�?!7����@D@)�O9&��?1�ڻ���5@:Preprocessing2U
Iterator::Model::ParallelMapV2h�XR�>�?!��L#�2@)h�XR�>�?1��L#�2@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����݆?!��1��<"@)����݆?1��1��<"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip(�����?!�RF �M@)��;3��?1��-��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�=&R�͓?!q�0z�/@)8�k����?1�}�`ϳ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensork��=]}?!E/htWk@)k��=]}?1E/htWk@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�w�1!�?!k���?�1@)��I���b?1.��(-��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�43.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9T�@���@I�G��o�P@Q��2�o�=@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	f���8��?f���8��?!f���8��?      ��!       "	�闈�� @�闈�� @!�闈�� @*      ��!       2	Ƨ Ϡ�?Ƨ Ϡ�?!Ƨ Ϡ�?:	N�f�}@N�f�}@!N�f�}@B      ��!       J	V�����?V�����?!V�����?R      ��!       Z	V�����?V�����?!V�����?b      ��!       JGPUYT�@���@b q�G��o�P@y��2�o�=@