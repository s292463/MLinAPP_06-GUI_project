	?`6@?`6@!?`6@	8p??R@8p??R@!8p??R@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?`6@??????1????@A?_w??ē?I)_?B&@Yp^??jG??rEagerKernelExecute 0*	??~j??d@2F
Iterator::Model?j-?B;??!??ױ??F@)	kc섗??1??c,?1=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?dT8??!<!?? ?@)?Z?[!???1Ru?V?:@:Preprocessing2U
Iterator::Model::ParallelMapV2_??W???!??K7?w0@)_??W???1??K7?w0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????˓?!?P@?+?'@)????˓?1?P@?+?'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?,C????!1(Na+K@)???W???1??K??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate؛?????!?ZmnX0@)??<?~?1[]?xba@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor+?MF?a|?!?s????@)+?MF?a|?1?s????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Lh?XR??!??????1@)}(Ff?1??u?Gq??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?29.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no98p??R@I?gs?? G@QH1??0?I@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????????????!??????      ??!       "	????@????@!????@*      ??!       2	?_w??ē??_w??ē?!?_w??ē?:	)_?B&@)_?B&@!)_?B&@B      ??!       J	p^??jG??p^??jG??!p^??jG??R      ??!       Z	p^??jG??p^??jG??!p^??jG??b      ??!       JGPUY8p??R@b q?gs?? G@yH1??0?I@