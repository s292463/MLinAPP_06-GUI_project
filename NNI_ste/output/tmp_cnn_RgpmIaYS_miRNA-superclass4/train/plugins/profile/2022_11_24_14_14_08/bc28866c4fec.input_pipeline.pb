	^-wfR"@^-wfR"@!^-wfR"@	me?k????me?k????!me?k????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL^-wfR"@Ō?? ??1"???@Aۅ?:????Iۊ?e??@Yg???u??rEagerKernelExecute 0*	^?I?d@2F
Iterator::ModelO??'????!>??G@)??*????1ٶinb'=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat&?\R?ݤ?!+ ?t?q8@)????u???1??3 ??4@:Preprocessing2U
Iterator::Model::ParallelMapV2k'JB"??!?????1@)k'JB"??1?????1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?G??'???!?MN???J@)f?"?ϙ?1Ml??(=.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??$??}??!W(ݜM?@)??$??}??1W(ݜM?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?#??ŋ??!i?	&?=)@)l??g????1~86?:?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??i??y?!???j@)??i??y?1???j@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?/fKVE??!a???n,@)?a0??e?1???랉??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?45.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9le?k????I
>?k.QP@Q??
?vh@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ō?? ??Ō?? ??!Ō?? ??      ??!       "	"???@"???@!"???@*      ??!       2	ۅ?:????ۅ?:????!ۅ?:????:	ۊ?e??@ۊ?e??@!ۊ?e??@B      ??!       J	g???u??g???u??!g???u??R      ??!       Z	g???u??g???u??!g???u??b      ??!       JGPUYle?k????b q
>?k.QP@y??
?vh@@