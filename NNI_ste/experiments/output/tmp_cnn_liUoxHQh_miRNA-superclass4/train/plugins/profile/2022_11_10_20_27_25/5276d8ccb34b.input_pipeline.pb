	Ox	N}?!@Ox	N}?!@!Ox	N}?!@	???^{?@???^{?@!???^{?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLOx	N}?!@od????1?j??w@A??0???I=?Ƃ@Y^-wf????rEagerKernelExecute 0*	?p=
?[^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????[??!d???,A@)f??ᔹ??1Ɠc?<@:Preprocessing2F
Iterator::Model!?????!?ۦ??C@)eM.????1?0??5@:Preprocessing2U
Iterator::Model::ParallelMapV2?'??9x??!x?5?2@)?'??9x??1x?5?2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??K?'??!??L?"@)??K?'??1??L?"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?V?????!?J??N1@)???W;???1?Bxqm@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?i?:Ȳ?!Z$Y?:5N@)?A?F????1.??E8@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?!q??}?!????^@)?!q??}?1????^@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?? ????!"a??3@)????3?`?1?f??zI??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?49.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???^{?@I?<?T?~Q@Q5;?@?%:@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	od????od????!od????      ??!       "	?j??w@?j??w@!?j??w@*      ??!       2	??0?????0???!??0???:	=?Ƃ@=?Ƃ@!=?Ƃ@B      ??!       J	^-wf????^-wf????!^-wf????R      ??!       Z	^-wf????^-wf????!^-wf????b      ??!       JGPUY???^{?@b q?<?T?~Q@y5;?@?%:@