	$
-???@$
-???@!$
-???@	KO?????KO?????!KO?????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL$
-???@?Բ?~??1??R????Aƾd????I?f???	@Y~?.rO??rEagerKernelExecute 0*??S??S^@)      =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??n????!??$??@@)?O ?Ȓ??1K??YK<@:Preprocessing2F
Iterator::Model?Y-??D??!?BJ?%E@)?j???t??1AK\???7@:Preprocessing2U
Iterator::Model::ParallelMapV2?Hh˹??!f:8???2@)?Hh˹??1f:8???2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??M񸨆?!n?@?="@)??M񸨆?1n?@?="@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateh!?˛??!??l6?/@);??Tގ??1r?ݨ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????????!-???R?L@)O#-??#|?1?u??<?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor8?q???{?!?:A??B@)8?q???{?1?:A??B@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?k?6???!???{?.2@)?R????g?1?g???-@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?47.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9KO?????I???DOQ@QӠ???&=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Բ?~???Բ?~??!?Բ?~??      ??!       "	??R??????R????!??R????*      ??!       2	ƾd????ƾd????!ƾd????:	?f???	@?f???	@!?f???	@B      ??!       J	~?.rO??~?.rO??!~?.rO??R      ??!       Z	~?.rO??~?.rO??!~?.rO??b      ??!       JGPUYKO?????b q???DOQ@yӠ???&=@