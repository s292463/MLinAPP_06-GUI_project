	0??L??@0??L??@!0??L??@	??\Pn@??\Pn@!??\Pn@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL0??L??@?0?:9C??1&???@Ad??Tkav?I ??	?9 @Y?)V????rEagerKernelExecute 0*	㥛?  d@2F
Iterator::Model???^??!D??/+?G@):?w????1??a???@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?????	??!?YS <@)??????1?#?$?F7@:Preprocessing2U
Iterator::Model::ParallelMapV2m?OT6???!?????V/@)m?OT6???1?????V/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateP÷?n???!L?Ӧ?0@)??!o????1T?fk?%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?+?j???!???~??@)?+?j???1???~??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??j+????!?jZ??[J@)?}?<???17????!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorP?????!?ظ?'a@)P?????1?ظ?'a@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?@-Ӟ?!3?2@)?ꫫ?h?1??X?(??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?27.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??\Pn@I???}?a?@Q?Z.FP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?0?:9C???0?:9C??!?0?:9C??      ??!       "	&???@&???@!&???@*      ??!       2	d??Tkav?d??Tkav?!d??Tkav?:	 ??	?9 @ ??	?9 @! ??	?9 @B      ??!       J	?)V?????)V????!?)V????R      ??!       Z	?)V?????)V????!?)V????b      ??!       JGPUY??\Pn@b q???}?a?@y?Z.FP@