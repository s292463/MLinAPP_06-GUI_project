	??{?@??{?@!??{?@	KM?6M@KM?6M@!KM?6M@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??{?@R?Hڍ>??1'?;E@A?H??Q,??IC?l????Y4???????rEagerKernelExecute 0*	?t??g@2F
Iterator::Modelh^????!ȶ?"??F@)%Z?xZ~??1?-F?@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat7?h?????!?>+e>@)? :vP??1g,???9@:Preprocessing2U
Iterator::Model::ParallelMapV2
pUj??!????'@)
pUj??1????'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??W??͔?!.?뛟Y%@)??W??͔?1.?뛟Y%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateN'??rJ??!??t??0@)?o?N\???1???,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????j???!8I9?tRK@)?I?5?o??1:?Du@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorӢ>?6??! ?[Pݩ@)Ӣ>?6??1 ?[Pݩ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??????!?w4+!~2@)???2??k?1?????a??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?24.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9KM?6M@I??0?D@Q?zDK@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	R?Hڍ>??R?Hڍ>??!R?Hڍ>??      ??!       "	'?;E@'?;E@!'?;E@*      ??!       2	?H??Q,???H??Q,??!?H??Q,??:	C?l????C?l????!C?l????B      ??!       J	4???????4???????!4???????R      ??!       Z	4???????4???????!4???????b      ??!       JGPUYKM?6M@b q??0?D@y?zDK@