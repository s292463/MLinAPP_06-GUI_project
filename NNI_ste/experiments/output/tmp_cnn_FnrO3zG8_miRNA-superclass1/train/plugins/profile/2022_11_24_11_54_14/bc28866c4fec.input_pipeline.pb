	?p??@@?p??@@!?p??@@	fצ???fצ???!fצ???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?p??@@?|#?g???1\qqTn???Ad?g^???I???Q?=@YSB??^~??rEagerKernelExecute 0*	???Mb@c@2F
Iterator::Model?o???߱?!<b?F@)???'??1?v?!?>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??M(??!?
?9@)R?U򱻠?1M%z?m85@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??|?͍??!ĝ0{UK@)1?Tm7???1??@?.@:Preprocessing2U
Iterator::Model::ParallelMapV2R_?vj.??!?kQF?e-@)R_?vj.??1?kQF?e-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice!?????!?y?H?@)!?????1?y?H?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateR(__???!w4K1?'@)?@?"??1??Œ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?iT?d{?!H?3=?^@)?iT?d{?1H?3=?^@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapn??fc%??!4W??,@)??c> ?i?1"^ @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?89.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9eצ???I?&??$?W@Q?#3ظ\@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?|#?g????|#?g???!?|#?g???      ??!       "	\qqTn???\qqTn???!\qqTn???*      ??!       2	d?g^???d?g^???!d?g^???:	???Q?=@???Q?=@!???Q?=@B      ??!       J	SB??^~??SB??^~??!SB??^~??R      ??!       Z	SB??^~??SB??^~??!SB??^~??b      ??!       JGPUYeצ???b q?&??$?W@y?#3ظ\@