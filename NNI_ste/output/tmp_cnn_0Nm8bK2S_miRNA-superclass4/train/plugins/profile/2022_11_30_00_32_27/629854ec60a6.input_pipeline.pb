	?s}?3@?s}?3@!?s}?3@	׋???j??׋???j??!׋???j??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?s}?3@???tB??1?	j?1@IvöE????Y?}r 
??rEagerKernelExecute 0*	_??"?e@2F
Iterator::Model?t???l??!?7?H@)?%?<Y??1(?-??o@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??$Ί???!??????=@)?`obHN??1\????9@:Preprocessing2U
Iterator::Model::ParallelMapV2+???}??!??? ??0@)+???}??1??? ??0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice2?CP5??!X?+d@)2?CP5??1X?+d@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??1=a???!p????J+@)NA~6r݄?1?E?W?1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???H????!yX~?'I@)????3???1L???P?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor֪]?z?!??$??@)֪]?z?1??$??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg??e???!@?dfG/@)?5Φ#?k?1z??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9׋???j??Ix??)FC'@QB????U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???tB?????tB??!???tB??      ??!       "	?	j?1@?	j?1@!?	j?1@*      ??!       2      ??!       :	vöE????vöE????!vöE????B      ??!       J	?}r 
???}r 
??!?}r 
??R      ??!       Z	?}r 
???}r 
??!?}r 
??b      ??!       JGPUY׋???j??b qx??)FC'@yB????U@