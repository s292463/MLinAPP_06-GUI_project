	=((E+?4@=((E+?4@!=((E+?4@      ??!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:=((E+?4@?CԷL??1lˀ??d2@I+j0???rEagerKernelExecute 0*	?/?$?u@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateq!??F???!?
??R?M@)DN_??,??1????:L@:Preprocessing2F
Iterator::Model?t_?l??!?y???5@)??"j?ϧ?1ބAu?*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatVn2???!?5/f4.@)?r?w????16)㵱0*@:Preprocessing2U
Iterator::Model::ParallelMapV2e??????!9n?? @)e??????19n?? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?2??։?!?#?s??@)?2??։?1?#?s??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??	ܺ???!?!?|?S@)L⬈????1[uՙ?'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??Dׅ|?!?b`?L??)??Dׅ|?1?b`?L??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?????5??!??GN@)t^c???j?1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?7.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?lR/?$@Qm?u?aV@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?CԷL???CԷL??!?CԷL??      ??!       "	lˀ??d2@lˀ??d2@!lˀ??d2@*      ??!       2      ??!       :	+j0???+j0???!+j0???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?lR/?$@ym?u?aV@