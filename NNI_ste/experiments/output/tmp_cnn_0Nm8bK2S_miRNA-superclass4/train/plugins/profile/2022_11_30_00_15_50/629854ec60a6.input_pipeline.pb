	DL?$z?!@DL?$z?!@!DL?$z?!@	??+]@??+]@!??+]@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCDL?$z?!@?y0H???1V???n?@I??X? @YR%?S;??rEagerKernelExecute 0*	?/?$rg@2F
Iterator::Model?O?????!u{' !?F@){JΉ=???1??2Բ?@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX???ާ??!?G?Ȩ?;@)^?c@?z??1?l??.s8@:Preprocessing2U
Iterator::Model::ParallelMapV2z?ؘ???!?*ӯ?(@)z?ؘ???1?*ӯ?(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?gyܝ??!L??`?&@)?gyܝ??1L??`?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??g?e??!?????|K@)Z???֑?1k?'e?"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?????u??!???7a0@)J|?????1e????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?w??Dgy?!n?m?s
@)?w??Dgy?1n?m?s
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??!6X8??!?9,?b?1@)I??Z??g?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?23.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??+]@Iȑ?BuU<@Q?<o?Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?y0H????y0H???!?y0H???      ??!       "	V???n?@V???n?@!V???n?@*      ??!       2      ??!       :	??X? @??X? @!??X? @B      ??!       J	R%?S;??R%?S;??!R%?S;??R      ??!       Z	R%?S;??R%?S;??!R%?S;??b      ??!       JGPUY??+]@b qȑ?BuU<@y?<o?Q@