	 ???Q?@ ???Q?@! ???Q?@	??!?Q@??!?Q@!??!?Q@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL ???Q?@?t??.???1?Z??D	@A???^(`??I??^[@Y?}V?)???rEagerKernelExecute 0*	?ʡE??c@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat`9B????!M???|?@@)??ڊ?e??1+????<@:Preprocessing2F
Iterator::Modelg?R@????!???s?D@)??zi? ??1?r??+M<@:Preprocessing2U
Iterator::Model::ParallelMapV2?}U.T???!ʑ?tx+@)?}U.T???1ʑ?tx+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??5?e???!)"??M@)?+?j???1w???%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice^f?(???!?'???m@)^f?(???1?'???m@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateF?-t%??!?U?b0?)@)?@?"??1i?j=?D@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorY?;ۣ7|?!??ߛ?[@)Y?;ۣ7|?1??ߛ?[@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap1е/???!?hR??-@)?R?d?1њ ?A???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?49.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??!?Q@I?QE4x,L@Q7??k?D@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?t??.????t??.???!?t??.???      ??!       "	?Z??D	@?Z??D	@!?Z??D	@*      ??!       2	???^(`?????^(`??!???^(`??:	??^[@??^[@!??^[@B      ??!       J	?}V?)????}V?)???!?}V?)???R      ??!       Z	?}V?)????}V?)???!?}V?)???b      ??!       JGPUY??!?Q@b q?QE4x,L@y7??k?D@