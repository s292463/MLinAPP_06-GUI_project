	??I?!@??I?!@!??I?!@	??t? @??t? @!??t? @"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??I?!@???y?C??19EGr???A??ߠ????Iș&l??@Y3NCT????rEagerKernelExecute 0*	????K?^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?/EHݦ?!????)DB@)j.7갢?1?bBSL?=@:Preprocessing2F
Iterator::Model+??$Ί??!;?N?C@)>????1{?	?C7@:Preprocessing2U
Iterator::Model::ParallelMapV2S8?????!??h?/?/@)S8?????1??h?/?/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicef??
???!?ʚ9r"@)f??
???1?ʚ9r"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipK?????!???!?dN@)-?}́?1?ә?Pq@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorup?x???!<?,b?@)up?x???1<?,b?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateΈ?????!???n.@)o??m?~?1io?c?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??Z}u??!????$1@)?Ia??Lc?1Y??̃???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?66.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??t? @I&3:??V@Q????#@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???y?C?????y?C??!???y?C??      ??!       "	9EGr???9EGr???!9EGr???*      ??!       2	??ߠ??????ߠ????!??ߠ????:	ș&l??@ș&l??@!ș&l??@B      ??!       J	3NCT????3NCT????!3NCT????R      ??!       Z	3NCT????3NCT????!3NCT????b      ??!       JGPUY??t? @b q&3:??V@y????#@