	@?߾?@@?߾?@!@?߾?@	???c?R@???c?R@!???c?R@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL@?߾?@?{?Y?H??1c+hZb?@A?[?tYL??I?s֧S@Y???ި??rEagerKernelExecute 0*	hffff"f@2F
Iterator::ModelV?@?)V??!#?K??G@)?x]?`??1??PNA@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate>"?D???!???pgvB@)o)狽??1C:ٛx9@:Preprocessing2U
Iterator::Model::ParallelMapV2!w?(???!?uϼ??(@)!w?(???1?uϼ??(@:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?"k????!?,???$@)?"k????1?,???$@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???Y????!Y;???$@)?!?k^Չ?1nT??~@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipѯ?????!?l??EwJ@)????~?11ܢ2?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/???0x?!?D?<?
@)/???0x?1?D?<?
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??LLb??!?Z?jk,C@)ѭ????d?1??9???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?8?Վ?\?!wZ?q%???)?8?Վ?\?1wZ?q%???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 17.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?37.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???c?R@I??1??K@Q?FRL??C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?{?Y?H???{?Y?H??!?{?Y?H??      ??!       "	c+hZb?@c+hZb?@!c+hZb?@*      ??!       2	?[?tYL???[?tYL??!?[?tYL??:	?s֧S@?s֧S@!?s֧S@B      ??!       J	???ި?????ި??!???ި??R      ??!       Z	???ި?????ި??!???ި??b      ??!       JGPUY???c?R@b q??1??K@y?FRL??C@