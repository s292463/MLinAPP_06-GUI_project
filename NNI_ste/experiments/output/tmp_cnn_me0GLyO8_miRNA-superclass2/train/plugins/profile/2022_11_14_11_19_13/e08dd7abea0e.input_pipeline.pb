	H¾??@H¾??@!H¾??@	??S??@??S??@!??S??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLH¾??@F?????1??:7m???AV????_??I_&???@Y5c?tv2??rEagerKernelExecute 0*	gfffff`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX?L??~??!?v%jW?@@)ZJ??P???1q>?c@<@:Preprocessing2F
Iterator::Model
,?)??!?]?ڕ?D@)???I???1ԮD?J49@:Preprocessing2U
Iterator::Model::ParallelMapV21(?hr1??!}????0@)1(?hr1??1}????0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice<???ܴ??!?v%jW"#@)<???ܴ??1?v%jW"#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatek?m?\p??!,Q???0@)???&?+??1jW?v%?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip |(ђ??!W?v%j#M@)??Û5x?1???|l@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??%?"|?!Q??+?@)??%?"|?1Q??+?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?UfJ?o??!?cp>?2@)??v?g?1ڕ?]??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?55.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??S??@I??װu|S@Q5?L%?1@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	F?????F?????!F?????      ??!       "	??:7m?????:7m???!??:7m???*      ??!       2	V????_??V????_??!V????_??:	_&???@_&???@!_&???@B      ??!       J	5c?tv2??5c?tv2??!5c?tv2??R      ??!       Z	5c?tv2??5c?tv2??!5c?tv2??b      ??!       JGPUY??S??@b q??װu|S@y5?L%?1@