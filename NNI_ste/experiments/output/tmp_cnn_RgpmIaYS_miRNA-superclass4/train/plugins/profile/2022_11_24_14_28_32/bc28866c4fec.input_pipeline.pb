	????53!@????53!@!????53!@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC????53!@ׄ?Ơ???11\ q7@A]~p>??I????!	@rEagerKernelExecute 0*	Zd;?'a@2F
Iterator::ModelG6u??! M?V?\H@)75?|?ݦ?1?7:E@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???#bJ??!??b
?<@)????Q??1?.BA?8@:Preprocessing2U
Iterator::Model::ParallelMapV2????7???!8L???.0@)????7???18L???.0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?Dg?E(??!?ʣ??@)?Dg?E(??1?ʣ??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??o?????!0^??,@)??x??M??1z?z|؟@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipj?t???!߲?e?I@)2??8*7??1?ZF[?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??s???w?!m???$?@)??s???w?1m???$?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa??????![???F0@)???Z(i?1r????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 17.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?36.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??j?GPK@QG?*??F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ׄ?Ơ???ׄ?Ơ???!ׄ?Ơ???      ??!       "	1\ q7@1\ q7@!1\ q7@*      ??!       2	]~p>??]~p>??!]~p>??:	????!	@????!	@!????!	@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??j?GPK@yG?*??F@