	????%@????%@!????%@	 ??Z??? ??Z???! ??Z???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL????%@y:W???1?????@A?Z?7?q??I?Z?7?q??Y2??%????rEagerKernelExecute 0*	O??n?`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?U?&???!TPr????@)????ɍ??1`???M;@:Preprocessing2F
Iterator::Model???ͪ?!>t???C@)A?º???1?????6@:Preprocessing2U
Iterator::Model::ParallelMapV2?C?|???!?nd\$?0@)?C?|???1?nd\$?0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice,??d??!1?]??.@),??d??11?]??.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??????!???CGN@)|C??up??1?$??-1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????"2??!?6?|?4@)쉮?8?1Z!E ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorԷ???x?!??g,@)Է???x?1??g,@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????˞?!?M0٨6@)a??>??d?1?p?;˛??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?16.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9 ??Z???I?????@@QNi?PP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	y:W???y:W???!y:W???      ??!       "	?????@?????@!?????@*      ??!       2	?Z?7?q???Z?7?q??!?Z?7?q??:	?Z?7?q???Z?7?q??!?Z?7?q??B      ??!       J	2??%????2??%????!2??%????R      ??!       Z	2??%????2??%????!2??%????b      ??!       JGPUY ??Z???b q?????@@yNi?PP@