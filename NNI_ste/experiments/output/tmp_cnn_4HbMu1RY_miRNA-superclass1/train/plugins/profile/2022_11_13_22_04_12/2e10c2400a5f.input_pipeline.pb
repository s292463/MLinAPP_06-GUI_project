	
h"lxZ@
h"lxZ@!
h"lxZ@	���7!z�?���7!z�?!���7!z�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL
h"lxZ@YO���*�?1U���i�?A!Z+��?I[�[!��@Y>��@�?rEagerKernelExecute 0*	��� �`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?����?!2Pͳ@@)-ͭVc�?1���۾�:@:Preprocessing2F
Iterator::Model�����?!����MC@)���[�?1��d��5@:Preprocessing2U
Iterator::Model::ParallelMapV2���.��?!5{�	p0@)���.��?15{�	p0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�<e5]O�?!�.�e�.@)�<e5]O�?1�.�e�.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��F̜?!�rI��5@)U����?1s.����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�L��O�?!u"a��N@)�{b�*?1�H�85�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����oa}?!�%��f@)����oa}?1�%��f@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���͎T�?!��=�/�7@)y���ABd?1���h��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 10.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�58.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���7!z�?I	�۪OQ@Q!�5~��<@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	YO���*�?YO���*�?!YO���*�?      ��!       "	U���i�?U���i�?!U���i�?*      ��!       2	!Z+��?!Z+��?!!Z+��?:	[�[!��@[�[!��@![�[!��@B      ��!       J	>��@�?>��@�?!>��@�?R      ��!       Z	>��@�?>��@�?!>��@�?b      ��!       JGPUY���7!z�?b q	�۪OQ@y!�5~��<@