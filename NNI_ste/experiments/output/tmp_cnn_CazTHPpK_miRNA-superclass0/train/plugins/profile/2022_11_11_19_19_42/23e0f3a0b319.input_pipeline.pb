	8?????!@8?????!@!8?????!@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC8?????!@?4?BXM??1?Sͬ???AOWw,?I??Iq?q?t?@rEagerKernelExecute 0*	?$??;d@2F
Iterator::Model???"1A??!?Q?0?D@)???I??1f?1F7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_9?????!^?,|859@)?p?????1??d!??5@:Preprocessing2U
Iterator::Model::ParallelMapV26׆?q??!??V]^2@)6׆?q??1??V]^2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceP??0{ٖ?!??|?V?+@)P??0{ٖ?1??|?V?+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?#EdX??!Zw????9@)?&OYMד?1??k??'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipGY???.??!?;??-M@)?.l?V^??1?ƍ? *@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorû\?wbv?!?$>ֺ@)û\?wbv?1?$>ֺ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapo?ꐛ???!?V??$?;@)?~j?t?h?1?-?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 22.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?59.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??V??mT@Q˹??AI2@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?4?BXM???4?BXM??!?4?BXM??      ??!       "	?Sͬ????Sͬ???!?Sͬ???*      ??!       2	OWw,?I??OWw,?I??!OWw,?I??:	q?q?t?@q?q?t?@!q?q?t?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??V??mT@y˹??AI2@