	?z?p?2!@?z?p?2!@!?z?p?2!@	??\?? @??\?? @!??\?? @"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?z?p?2!@?U1?>??19}=_#@A???jה?It^c?????Ys+??X???rEagerKernelExecute 0*	?Zd;?t@2Z
#Iterator::Model::ParallelMapV2::Zip???8???!?K??I?S@)N'??rJ??1?1??cJ@:Preprocessing2F
Iterator::Model?;??ؖ??!fт???4@)??n???1?۝?,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?R\U?]??!?R?.@)???;??1??$'*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???'???!@?<u?@)???'???1@?<u?@:Preprocessing2U
Iterator::Model::ParallelMapV2??r??ږ?!DBU?Y@)??r??ږ?1DBU?Y@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateo+?6+??!:??%LS$@)?i??????1??%FH@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?7?{?5z?!???????)?7?{?5z?1???????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap}?:??!?????&@)e?z?Fwp?1???
f~??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 13.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?17.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??\?? @I,????>@Q??-+?P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?U1?>???U1?>??!?U1?>??      ??!       "	9}=_#@9}=_#@!9}=_#@*      ??!       2	???jה????jה?!???jה?:	t^c?????t^c?????!t^c?????B      ??!       J	s+??X???s+??X???!s+??X???R      ??!       Z	s+??X???s+??X???!s+??X???b      ??!       JGPUY??\?? @b q,????>@y??-+?P@