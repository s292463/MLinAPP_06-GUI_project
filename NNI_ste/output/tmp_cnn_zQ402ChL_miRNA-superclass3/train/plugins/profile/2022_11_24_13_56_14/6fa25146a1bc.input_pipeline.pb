	׾?^?? @׾?^?? @!׾?^?? @      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC׾?^?? @߉Y/?@1???x????A??bc^G??I]?@??@rEagerKernelExecute 0*	1?Z?c@2F
Iterator::ModelA??4F???!?ʃ@Y J@)??????1?P4=??B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???1ZG??!?Z??r:@)???WW??1tW)f6@:Preprocessing2U
Iterator::Model::ParallelMapV2??s??Ɨ?!|?=ԍ-@)??s??Ɨ?1|?=ԍ-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice8????C??!^1b??f@)8????C??1^1b??f@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipc~nh?N??!5|???G@)C=}????18???B?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatez ???!??!RD???+@)??X ??1?r&a%?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorw?x?z?!?	
?2@)w?x?z?1?	
?2@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapv?1<????!??2}a?.@)??6?ُd?1d?rﶎ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 29.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?50.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI|B?p\%T@Q??=?j3@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	߉Y/?@߉Y/?@!߉Y/?@      ??!       "	???x???????x????!???x????*      ??!       2	??bc^G????bc^G??!??bc^G??:	]?@??@]?@??@!]?@??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q|B?p\%T@y??=?j3@