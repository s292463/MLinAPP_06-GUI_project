	?Ȱ?7?@?Ȱ?7?@!?Ȱ?7?@	FNo????FNo????!FNo????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?Ȱ?7?@b.?? @1(??Z&???A? @????I?}?k?L@Y?C9Ѯ??rEagerKernelExecute 0*	bX9?hd@2F
Iterator::Modele????c??!`?6??E@)???8?j??1??-?g>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat%??1 ??!??2v9@@)??^b,ӧ?1??LD?<@:Preprocessing2U
Iterator::Model::ParallelMapV2[??ù??!?_??p/+@)[??ù??1?_??p/+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?yS?
c??!?~?qa0@)??%ǝґ?1L???Q%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice7l[?? ??!?^????@)7l[?? ??1?^????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipղ??Hh??!??U?5 L@)?4)?^??1??(???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?Y?b+hz?!L`?ɖ@)?Y?b+hz?1L`?ɖ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????P1??!????2@)Qf?L2rf?1?7?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 28.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?52.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9FNo????I7?hFT@Qs???(1@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	b.?? @b.?? @!b.?? @      ??!       "	(??Z&???(??Z&???!(??Z&???*      ??!       2	? @????? @????!? @????:	?}?k?L@?}?k?L@!?}?k?L@B      ??!       J	?C9Ѯ???C9Ѯ??!?C9Ѯ??R      ??!       Z	?C9Ѯ???C9Ѯ??!?C9Ѯ??b      ??!       JGPUYFNo????b q7?hFT@ys???(1@