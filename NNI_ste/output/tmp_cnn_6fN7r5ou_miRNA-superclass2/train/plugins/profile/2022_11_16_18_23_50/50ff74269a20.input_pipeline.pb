	>�#d�@>�#d�@!>�#d�@	��iz��
@��iz��
@!��iz��
@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL>�#d�@w.�����?1.�ED1y�?A��w��?Iڮ���
@Y@�&M���?rEagerKernelExecute 0*	�����f@2F
Iterator::Model~�,��?!cĎ�O�F@)4I,)w��?1�U����>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatF	�=b�?!�Ί�6@)|�5Z��?1����2@:Preprocessing2U
Iterator::Model::ParallelMapV2�c��1�?!Mf ��%.@)�c��1�?1Mf ��%.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipTrN�}�?!�;qU�&K@)K�.��"�?1t��S�,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliced\qqTn�?!�)��n$@)d\qqTn�?1�)��n$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�J���>�?!*�|'wP/@)�A�L��?1V��0�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorS�!�uq{?!;Є��l@)S�!�uq{?1;Є��l@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapy�Z�?!��v�71@)?$D��f?1�<�%���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 20.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�48.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��iz��
@I�}@�QQ@Q�I�.�c;@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	w.�����?w.�����?!w.�����?      ��!       "	.�ED1y�?.�ED1y�?!.�ED1y�?*      ��!       2	��w��?��w��?!��w��?:	ڮ���
@ڮ���
@!ڮ���
@B      ��!       J	@�&M���?@�&M���?!@�&M���?R      ��!       Z	@�&M���?@�&M���?!@�&M���?b      ��!       JGPUY��iz��
@b q�}@�QQ@y�I�.�c;@