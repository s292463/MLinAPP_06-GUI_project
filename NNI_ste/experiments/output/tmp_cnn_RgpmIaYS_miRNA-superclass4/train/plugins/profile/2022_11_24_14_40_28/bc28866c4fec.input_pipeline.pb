	?hV?I0@?hV?I0@!?hV?I0@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?hV?I0@1Bx?q???1{?f?l?)@Aͮ{+??Ibg
??X??rEagerKernelExecute 0*	? ?rh?b@2F
Iterator::Modelv4?????!?Es?8G@)3???yS??1/?7?9?@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?&??b??!CX??Z<@)????O???1%t?L??7@:Preprocessing2U
Iterator::Model::ParallelMapV2s	????!M?boe*@)s	????1M?boe*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??{?专?!??/]?)@)??{?专?1??/]?)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateA	]ޜ?!?A̠n?2@)?\??J??1??$ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???f??!>???j?J@)????y}?1??Q$IX@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??	?y{?!y?K???@)??	?y{?1y?K???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapw.???v??!??$h?4@)??.?d?19?՗???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 10.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?10.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIŞ?-5@Q?N?
??S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	1Bx?q???1Bx?q???!1Bx?q???      ??!       "	{?f?l?)@{?f?l?)@!{?f?l?)@*      ??!       2	ͮ{+??ͮ{+??!ͮ{+??:	bg
??X??bg
??X??!bg
??X??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qŞ?-5@y?N?
??S@