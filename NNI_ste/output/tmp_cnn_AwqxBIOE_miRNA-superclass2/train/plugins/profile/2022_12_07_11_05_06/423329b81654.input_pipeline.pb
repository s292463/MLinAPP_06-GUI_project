	??8Q @??8Q @!??8Q @	.???f@.???f@!.???f@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??8Q @|??8G??13p@KW@A?h㈵?T?I??Dg?%@Y??ʡE??rEagerKernelExecute 0*	m??ʫt@2U
Iterator::Model::ParallelMapV2b??U???!NB????K@)b??U???1NB????K@:Preprocessing2F
Iterator::Model????/??!$? ?jR@)?-?l?I??1??:???1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?f???§?!&kj,@)??X????1?U??M?'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice<?$???!???NnP@)<?$???1???NnP@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateH¾?D???!???Ü?"@)VW@܅?1??r??	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorOt	?~?!??6?@)Ot	?~?1??6?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip*???K??!o????U:@)W?}W?{?1倳?c? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapհ??T??!?C??Tx$@)?P?,i?1?ڕ?}???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?28.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9/???f@Ibz?4?	A@Q܌?n?/O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	|??8G??|??8G??!|??8G??      ??!       "	3p@KW@3p@KW@!3p@KW@*      ??!       2	?h㈵?T??h㈵?T?!?h㈵?T?:	??Dg?%@??Dg?%@!??Dg?%@B      ??!       J	??ʡE????ʡE??!??ʡE??R      ??!       Z	??ʡE????ʡE??!??ʡE??b      ??!       JGPUY/???f@b qbz?4?	A@y܌?n?/O@