	����0�@����0�@!����0�@	�L�@�L�@!�L�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL����0�@Ӣ>���?1�V$&�!	@A�F���?IX�vMH+@Y��n���?rEagerKernelExecute 0*	�C�l�g`@2F
Iterator::Modelиp $�?!������G@)DL�$z�?1U)��f?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���^��?!�iެC=@)�>���ʟ?1ã'ȧ7@:Preprocessing2U
Iterator::Model::ParallelMapV2�J�8���?!�1�L�Y0@)�J�8���?1�1�L�Y0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceߣ�z��?!g�m*)!!@)ߣ�z��?1g�m*)!!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��A$C��?!hR9 J@)�eO�s�?1v�T�i{@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�����?!��uhCO-@)0��L�^�?1��|4\@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�qS��|?!��@)�qS��|?1��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap|,}���?!��>��1@)���$xCj?1�� ��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 15.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�38.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�L�@I�ۉ���J@Q<���W�E@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Ӣ>���?Ӣ>���?!Ӣ>���?      ��!       "	�V$&�!	@�V$&�!	@!�V$&�!	@*      ��!       2	�F���?�F���?!�F���?:	X�vMH+@X�vMH+@!X�vMH+@B      ��!       J	��n���?��n���?!��n���?R      ��!       Z	��n���?��n���?!��n���?b      ��!       JGPUY�L�@b q�ۉ���J@y<���W�E@