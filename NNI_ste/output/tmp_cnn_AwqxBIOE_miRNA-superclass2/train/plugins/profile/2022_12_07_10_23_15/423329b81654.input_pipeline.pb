	?2?g?@?2?g?@!?2?g?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?2?g?@?HLP?7??1g??67?@A???????I??k??@rEagerKernelExecute 0*	a??"?5c@2F
Iterator::Model?3/??w??!xx????D@)x` ?C???1???	?<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?A|`???!{?014@@)ӅX???1U?	t<@:Preprocessing2U
Iterator::Model::ParallelMapV29?Z?̔?!???Wo*@)9?Z?̔?1???Wo*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice3???VC??!z?o?5'@)3???VC??1z?o?5'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?K?[?߶?!??+%M@)??????1{q{??	@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?F?g?u??!?ݖs1@)?x?'e??1?2|?`@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor6t??Pn{?!_?ScAn@)6t??Pn{?1_?ScAn@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?ڋh;???!???K?y3@)s???i?1?@??5 @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?52.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???~n?N@Q{R??C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?HLP?7???HLP?7??!?HLP?7??      ??!       "	g??67?@g??67?@!g??67?@*      ??!       2	??????????????!???????:	??k??@??k??@!??k??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???~n?N@y{R??C@