	???v?@???v?@!???v?@	o?^Y?/"@o?^Y?/"@!o?^Y?/"@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???v?@h??WB??1t]?@?@A??CV??IB????@Y?"?????rEagerKernelExecute 0*	_??"??@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapo?1h??!?^`???K@)(v?U??1???QKJ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??x?@e??!7#J!?B@)1[?*????1?/?zTA@:Preprocessing2F
Iterator::Model?cϞ??!????'@)?U??6o??1?????@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeattD?K?K??!?xȳ@)B??v?$??1I??Ak?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?VAt???!??J?5/??)r2q? ??1?C??T??:Preprocessing2U
Iterator::Model::ParallelMapV2l\??Ϝ??!?X???	??)l\??Ϝ??1?X???	??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????????!??L?C@)>???4`??1?]#?N???:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch???????!?a?*??)???????1?a?*??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????7??!<??%????)Hlw?}??1k?m?)???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??? ?y?!????2??)??? ?y?1????2??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice%?S;?t?!륞??5??)%?S;?t?1륞??5??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??2?68q?!s.??????)??2?68q?1s.??????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?????!?htA??)?m?f?1O??????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorG6u^?!??x ????)G6u^?1??x ????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?39.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t18.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9o?^Y?/"@I_????M@Q??=[@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	h??WB??h??WB??!h??WB??      ??!       "	t]?@?@t]?@?@!t]?@?@*      ??!       2	??CV????CV??!??CV??:	B????@B????@!B????@B      ??!       J	?"??????"?????!?"?????R      ??!       Z	?"??????"?????!?"?????b      ??!       JGPUYo?^Y?/"@b q_????M@y??=[@@