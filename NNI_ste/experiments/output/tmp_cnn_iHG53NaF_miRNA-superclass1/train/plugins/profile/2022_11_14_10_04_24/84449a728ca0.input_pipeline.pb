	�(�[Z-@�(�[Z-@!�(�[Z-@	�^��
@�^��
@!�^��
@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�(�[Z-@���I�?1uv28J@Ab�����?I��f�|@Yz ���!�?rEagerKernelExecute 0*	�t�.a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;:�Fv��?!�܉��>@)�C�b�?1ˇ$n� :@:Preprocessing2F
Iterator::ModelZ����?!��8RGD@) �����?1ald��9@:Preprocessing2U
Iterator::Model::ParallelMapV2t	4ؔ?!y���@�-@)t	4ؔ?1y���@�-@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�[���?!���Ǖ*@)�[���?1���Ǖ*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��9��?!P@lb4@)v�1<��?1���!^@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipwj.7�?!2�Nǭ�M@)��u�ݑ�?1�t���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��Tkaz?!MT��N�@)��Tkaz?1MT��N�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�r�]���?!��v�op6@)�ܚt["g?1�a9�p @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 4.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�50.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�^��
@I
�]�J�K@Qt�s7�D@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���I�?���I�?!���I�?      ��!       "	uv28J@uv28J@!uv28J@*      ��!       2	b�����?b�����?!b�����?:	��f�|@��f�|@!��f�|@B      ��!       J	z ���!�?z ���!�?!z ���!�?R      ��!       Z	z ���!�?z ���!�?!z ���!�?b      ��!       JGPUY�^��
@b q
�]�J�K@yt�s7�D@