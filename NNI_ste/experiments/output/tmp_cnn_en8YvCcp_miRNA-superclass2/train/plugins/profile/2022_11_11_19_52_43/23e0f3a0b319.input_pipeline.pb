	p?DIH?!@p?DIH?!@!p?DIH?!@	?"S?U!@?"S?U!@!?"S?U!@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLp?DIH?!@[????1?=~os @A4H?Sȕ??I_F????@Y?p?Qe???rEagerKernelExecute 0*	?MbX?l@2F
Iterator::Model?66;R}??!????P@)???>$??1 ???hK@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???(_Ъ?!?kdE?6@){.S????1?{??E5@:Preprocessing2U
Iterator::Model::ParallelMapV2{???ɚ?!???Xa?&@){???ɚ?1???Xa?&@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?W?ۼ??!?5{ʥ@)?ꫫ???1.??9?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?jH?c???!?????@@)????v?1??ۯGk@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???i?u?!?!I@)???i?u?1?!I@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap)??q??!?Q????7@)?X??+?d?1?`n??h??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?!9??U`?!?,??ϼ??)?!9??U`?1?,??ϼ??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?^?sa?W?!lo:????)?^?sa?W?1lo:????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?47.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t20.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?"S?U!@I??,	Q@Q?J?t07@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	[????[????![????      ??!       "	?=~os @?=~os @!?=~os @*      ??!       2	4H?Sȕ??4H?Sȕ??!4H?Sȕ??:	_F????@_F????@!_F????@B      ??!       J	?p?Qe????p?Qe???!?p?Qe???R      ??!       Z	?p?Qe????p?Qe???!?p?Qe???b      ??!       JGPUY?"S?U!@b q??,	Q@y?J?t07@