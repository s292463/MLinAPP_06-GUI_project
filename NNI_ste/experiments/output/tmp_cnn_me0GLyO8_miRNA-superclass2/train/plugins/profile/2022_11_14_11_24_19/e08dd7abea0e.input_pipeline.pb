	?ŊL?@?ŊL?@!?ŊL?@	?#?}?@?#?}?@!?#?}?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?ŊL?@~9?]???1??????A??N?`???Ish??|?@YK?ó??rEagerKernelExecute 0*	?rh???e@2F
Iterator::Model?Z{??B??!:"???E@)ڨN???1 yJϔ9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?P?f??!????d`:@)??w?'-??1P?|???6@:Preprocessing2U
Iterator::Model::ParallelMapV2????????!??A?s?1@)????????1??A?s?1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??ECƣ??!?y???C'@)??ECƣ??1?y???C'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?
??X??!???`^JL@)????cw??1?m???#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate F?6???!.??? ?2@)?x??M???1??Z	<@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorb?{???y?!?ڽ??@)b?{???y?1?ڽ??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapO!W?Y??!YR<?]\4@)????5"h?1?f??3??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 22.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?50.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?#?}?@I!??CI?R@Q??M+?6@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	~9?]???~9?]???!~9?]???      ??!       "	????????????!??????*      ??!       2	??N?`?????N?`???!??N?`???:	sh??|?@sh??|?@!sh??|?@B      ??!       J	K?ó??K?ó??!K?ó??R      ??!       Z	K?ó??K?ó??!K?ó??b      ??!       JGPUY?#?}?@b q!??CI?R@y??M+?6@