	lxz?,?@lxz?,?@!lxz?,?@	p?z??i@p?z??i@!p?z??i@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLlxz?,?@ kծ???16l??G@A}?b?: ??Iס????@Y.Y?&???rEagerKernelExecute 0*	?S㥛tc@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatCSv?A]??!?	ҵ??A@)y=????1Le	??>@:Preprocessing2F
Iterator::Model???l ??!?@M?}E@)???????1??B???;@:Preprocessing2U
Iterator::Model::ParallelMapV2?;??bF??!Y?|
:v.@)?;??bF??1Y?|
:v.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceиp $??!????+@)иp $??1????+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??vLݕ?!??^?o+@)?v1?t???1";?س@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???zܷ??!?}??!?L@)??????1,(ڇ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorS??.?}?!=???U?@)S??.?}?1=???U?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapz5@i?Q??!E??/_?.@)(?x?ߢc?1Ƒ????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?50.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9p?z??i@I?Y????M@QHS]W?B@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	 kծ??? kծ???! kծ???      ??!       "	6l??G@6l??G@!6l??G@*      ??!       2	}?b?: ??}?b?: ??!}?b?: ??:	ס????@ס????@!ס????@B      ??!       J	.Y?&???.Y?&???!.Y?&???R      ??!       Z	.Y?&???.Y?&???!.Y?&???b      ??!       JGPUYp?z??i@b q?Y????M@yHS]W?B@