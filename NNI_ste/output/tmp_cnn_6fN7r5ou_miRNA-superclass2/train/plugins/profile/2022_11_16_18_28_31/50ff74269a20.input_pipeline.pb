	?J?.??@?J?.??@!?J?.??@	?}Ǿ?3???}Ǿ?3??!?}Ǿ?3??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?J?.??@????????1'g(?x???A??WuV??I'????@Y??(?ָ?rEagerKernelExecute 0*		?Zd?b@2F
Iterator::Model ??G????!?????F@)K!?Ky??1?{]?G??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??\m????!?U???cA@)??լ3???1X?ʦ|?>@:Preprocessing2U
Iterator::Model::ParallelMapV2?)r??9??!bd?[5X+@)?)r??9??1bd?[5X+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???7?{??!?????@)???7?{??1?????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip? ?S?D??!iQ?NfK@)5??,??1????"i@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateb???u??! ?0O?)@)8gDio??1<Sή],@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensord!:?z?!>???\?@)d!:?z?1>???\?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaph?????!???|?V,@). ??Ld?1?pl?&??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 24.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?57.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?}Ǿ?3??I@b?u?T@Q ????j0@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????????????????!????????      ??!       "	'g(?x???'g(?x???!'g(?x???*      ??!       2	??WuV????WuV??!??WuV??:	'????@'????@!'????@B      ??!       J	??(?ָ???(?ָ?!??(?ָ?R      ??!       Z	??(?ָ???(?ָ?!??(?ָ?b      ??!       JGPUY?}Ǿ?3??b q@b?u?T@y ????j0@