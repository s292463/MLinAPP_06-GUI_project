	??*??O@??*??O@!??*??O@	?4 ???@?4 ???@!?4 ???@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??*??O@c|??l???1B?Ēr?@A???????I]7??V?@Y????a??rEagerKernelExecute 0*	?/?$~o@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateW?Y????!??_??nM@)`vOj??1?X+ K@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?V?????!???X??0@)??,`??1?q?}?},@:Preprocessing2F
Iterator::Modelӿ$?)???!???mM3@)I-?LN???1????l&@:Preprocessing2U
Iterator::Model::ParallelMapV2]R??ߔ?!?_*?. @)]R??ߔ?1?_*?. @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceu?? ???!˯???v@)u?? ???1˯???v@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipb?G??!????,T@)U??7???1!???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorݚt["|?!n????@)ݚt["|?1n????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?+?PO??!?kR?d?M@)?nf???d?1P?S0??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 17.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?38.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?4 ???@Iؚ?:??K@Q?a?*?GD@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	c|??l???c|??l???!c|??l???      ??!       "	B?Ēr?@B?Ēr?@!B?Ēr?@*      ??!       2	??????????????!???????:	]7??V?@]7??V?@!]7??V?@B      ??!       J	????a??????a??!????a??R      ??!       Z	????a??????a??!????a??b      ??!       JGPUY?4 ???@b qؚ?:??K@y?a?*?GD@