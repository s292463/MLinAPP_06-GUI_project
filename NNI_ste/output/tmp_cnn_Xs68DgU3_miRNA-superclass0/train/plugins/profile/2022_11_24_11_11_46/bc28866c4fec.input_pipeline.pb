	��{�@��{�@!��{�@	�[�؎	@�[�؎	@!�[�؎	@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��{�@�l˟�?1e���\= @AA��h:;�?I~�Ɍ��
@YF��\��?rEagerKernelExecute 0*	�K7�APs@2F
Iterator::Model�����?!� 'P�S@)�Za�^C�?1q+�{�P@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�N$�j�?!�$ V,@)�q�_!�?1��A��.(@:Preprocessing2U
Iterator::Model::ParallelMapV2�����?!�@�b#@)�����?1�@�b#@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����m��?!n@�uA	@)����m��?1n@�uA	@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�|]��t�?!0�r�@)�U���?1�����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�X�E��?!8|c���7@)ϣ�����?1M�/��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Iz?!Ay��� @)����Iz?1Ay��� @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�~NA~�?!�ϵ}�n@)cAJh?1B֏3��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 21.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�46.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�[�؎	@I�n�7�Q@Q��[F]<@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�l˟�?�l˟�?!�l˟�?      ��!       "	e���\= @e���\= @!e���\= @*      ��!       2	A��h:;�?A��h:;�?!A��h:;�?:	~�Ɍ��
@~�Ɍ��
@!~�Ɍ��
@B      ��!       J	F��\��?F��\��?!F��\��?R      ��!       Z	F��\��?F��\��?!F��\��?b      ��!       JGPUY�[�؎	@b q�n�7�Q@y��[F]<@