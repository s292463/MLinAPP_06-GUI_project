	ڨN�>@ڨN�>@!ڨN�>@	��	��@��	��@!��	��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLڨN�>@������?1���^�R@A�j�v/�?I�(\��@Y������?rEagerKernelExecute 0*	k�t�|a@2F
Iterator::Model��.m8,�?!���x�G@)��V��?1	%��1=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�.Ȗ�?!l?kD%>@)�\��'�?1���SgY9@:Preprocessing2U
Iterator::Model::ParallelMapV2�i��ߚ?!S���2@)�i��ߚ?1S���2@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice"4���߅?!�C��3�@)"4���߅?1�C��3�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��p�q��?!I���,@)Q�_�n�?1UN�C�"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�5?�Ң�?!Tx��J@)�U����?11�;@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorJ��	�y{?!@kI³.@)J��	�y{?1@kI³.@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�2�g�?!wkk/W0@)N�E� f?1ol�R��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 26.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�36.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��	��@IY�3j=O@Q�ҨT��@@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	������?������?!������?      ��!       "	���^�R@���^�R@!���^�R@*      ��!       2	�j�v/�?�j�v/�?!�j�v/�?:	�(\��@�(\��@!�(\��@B      ��!       J	������?������?!������?R      ��!       Z	������?������?!������?b      ��!       JGPUY��	��@b qY�3j=O@y�ҨT��@@