	?6?ُ?@?6?ُ?@!?6?ُ?@	VX??cf@VX??cf@!VX??cf@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?6?ُ?@w?h?hs??1?f?W%??AzUg????IDkE??@Y?6?~??rEagerKernelExecute 0*	?|?5^.a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatJ???nI??!'???P??@)??_̖???1X>?!?:@:Preprocessing2F
Iterator::Model?M?»\??!?9/?&D@)V???4???1g????9@:Preprocessing2U
Iterator::Model::ParallelMapV2|?ڥ???!=????.@)|?ڥ???1=????.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip+???}??!}???P?M@)]????ۑ?1/xpn`)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?t{Ic??!1?&ǂ?@)?t{Ic??11?&ǂ?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?}?
Ē?!????w?*@)6????$??1xck?l\@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor8fٓ??|?!:?AӾ?@)8fٓ??|?1:?AӾ?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???eN???!???Q3?.@)f??
?f?1?B? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 12.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?52.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9VX??cf@I^??$P@QuHqϒ?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	w?h?hs??w?h?hs??!w?h?hs??      ??!       "	?f?W%???f?W%??!?f?W%??*      ??!       2	zUg????zUg????!zUg????:	DkE??@DkE??@!DkE??@B      ??!       J	?6?~???6?~??!?6?~??R      ??!       Z	?6?~???6?~??!?6?~??b      ??!       JGPUYVX??cf@b q^??$P@yuHqϒ?>@