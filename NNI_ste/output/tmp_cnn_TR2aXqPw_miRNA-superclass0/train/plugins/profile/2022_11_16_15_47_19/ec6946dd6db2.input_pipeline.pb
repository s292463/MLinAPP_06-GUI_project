	ɯb??@ɯb??@!ɯb??@	m???[@m???[@!m???[@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLɯb??@1??PN???1??G?3 @A(?XQ?i??I??K?@Y???6???rEagerKernelExecute 0*	V-?yd@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ajK??!گ?^?6C@)?r?9>Z??1?{?*:?@@:Preprocessing2F
Iterator::Model??ʅʿ??!?u?PUB@)??C?l???1im?>??7@:Preprocessing2U
Iterator::Model::ParallelMapV2v4?????!h?????)@)v4?????1h?????)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?????!r?h???O@)?l??????1
??M??&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??\QJ??!?N??DE@)??\QJ??1?N??DE@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate؜?gB??!g????&@)??ܵ?|??1??Q?b?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF^???!????|@)F^???1????|@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Ֆ?!S?+?:+@)??Z(?l?1??V?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?44.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t17.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9m???[@I?3???O@Q?w?=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	1??PN???1??PN???!1??PN???      ??!       "	??G?3 @??G?3 @!??G?3 @*      ??!       2	(?XQ?i??(?XQ?i??!(?XQ?i??:	??K?@??K?@!??K?@B      ??!       J	???6??????6???!???6???R      ??!       Z	???6??????6???!???6???b      ??!       JGPUYm???[@b q?3???O@y?w?=@