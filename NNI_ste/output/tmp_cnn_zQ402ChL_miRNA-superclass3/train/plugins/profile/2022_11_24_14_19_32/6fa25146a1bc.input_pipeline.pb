	?^a??`@?^a??`@!?^a??`@	?x =Q? @?x =Q? @!?x =Q? @"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?^a??`@@???|??1?_=?[???A?EB[Υ??I?ǵ?bL@Y??%:?,??rEagerKernelExecute 0*	??v??Rc@2F
Iterator::ModelལƄ???!
???O?H@)?Ǚ&l???1??8?t??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatj?t???!LuA'd9@)?(?A&??1???y?5@:Preprocessing2U
Iterator::Model::ParallelMapV2h[?:???!???\*?1@)h[?:???1???\*?1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip? ??*???!?>}^?=I@)??OI??1?n	??#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?h?x?J??!g???@)?h?x?J??1g???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??$xC??!?v???*@)?@?w????1W?sJ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorAJ?i?w?!p??M@)AJ?i?w?1p??M@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?d??!??_"?j.@)\??b??g?1?_9??	??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 24.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?53.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?x =Q? @I.H????S@Q.?&%ΰ3@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	@???|??@???|??!@???|??      ??!       "	?_=?[????_=?[???!?_=?[???*      ??!       2	?EB[Υ???EB[Υ??!?EB[Υ??:	?ǵ?bL@?ǵ?bL@!?ǵ?bL@B      ??!       J	??%:?,????%:?,??!??%:?,??R      ??!       Z	??%:?,????%:?,??!??%:?,??b      ??!       JGPUY?x =Q? @b q.H????S@y.?&%ΰ3@