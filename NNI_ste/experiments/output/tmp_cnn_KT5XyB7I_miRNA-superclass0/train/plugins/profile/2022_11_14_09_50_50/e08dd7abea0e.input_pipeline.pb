	���J�@���J�@!���J�@	>=Fli/@>=Fli/@!>=Fli/@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL���J�@��ek}�?1Ϡ��K@A6���Ĕ?IW��m� @Yxe����?rEagerKernelExecute 0*	�z�Gu@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��� ��?!oΚ��M@)��� ��?1oΚ��M@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����&�?!1f�v�+@)dT8��?1�8C/��'@:Preprocessing2F
Iterator::Model8-x�W��?!�ޢ��1@)��8�j��?1���R��&@:Preprocessing2U
Iterator::Model::ParallelMapV21�~�٭�?!���)@)1�~�٭�?1���)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate:vP��?!�[�q��O@)z7e�?1I�^n��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��7����?!`H��T@)���'�?1Tӥc�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�D�$}?!���E� @)�D�$}?1���E� @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��Ӻ�?!��A�8P@)-|}�K�p?1O9�|#�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 20.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�28.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9>=Fli/@I���{��H@Q���{]H@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��ek}�?��ek}�?!��ek}�?      ��!       "	Ϡ��K@Ϡ��K@!Ϡ��K@*      ��!       2	6���Ĕ?6���Ĕ?!6���Ĕ?:	W��m� @W��m� @!W��m� @B      ��!       J	xe����?xe����?!xe����?R      ��!       Z	xe����?xe����?!xe����?b      ��!       JGPUY>=Fli/@b q���{��H@y���{]H@