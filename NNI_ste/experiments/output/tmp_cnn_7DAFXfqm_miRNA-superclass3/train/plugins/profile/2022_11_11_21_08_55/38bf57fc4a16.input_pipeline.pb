	?);??n@?);??n@!?);??n@	8Aj)?V@8Aj)?V@!8Aj)?V@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?);??n@??"?S??1f-?????AI?L?????I??DK?
@Yd??????rEagerKernelExecute 0*	?A`?к_@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;?s??q??!&&???@@)G?J????1F?S%;@:Preprocessing2F
Iterator::Model??R$_	??!Tl!??E@)1a4+ۇ??1k`???5@:Preprocessing2U
Iterator::Model::ParallelMapV2*q㊛?!??w?715@)*q㊛?1??w?715@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceap??/??!@?u_?!@)ap??/??1@?u_?!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?P?f??!T??.@)???,????1)$???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?P?y??!????umL@)???S? ??1??gq?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?Oq~?!Ԝx=l@)?Oq~?1Ԝx=l@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapk?MG ??!??o???1@)! _B?g?1?UãV@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?50.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no97Aj)?V@I̲??E?Q@Q???/\9@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??"?S????"?S??!??"?S??      ??!       "	f-?????f-?????!f-?????*      ??!       2	I?L?????I?L?????!I?L?????:	??DK?
@??DK?
@!??DK?
@B      ??!       J	d??????d??????!d??????R      ??!       Z	d??????d??????!d??????b      ??!       JGPUY7Aj)?V@b q̲??E?Q@y???/\9@