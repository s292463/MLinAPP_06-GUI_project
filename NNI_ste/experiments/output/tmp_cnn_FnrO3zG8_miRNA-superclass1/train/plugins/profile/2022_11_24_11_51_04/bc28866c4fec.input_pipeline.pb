	{???wg@{???wg@!{???wg@	?R?i@?R?i@!?R?i@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL{???wg@/??[<<??1B]¡? @AC??3??I??3ڪD	@Y????2??rEagerKernelExecute 0*	i??|???@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???/??!L??G??L@)?˛õ???1????lK@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(,???)??!????LB>@)oB@????1?G???:@:Preprocessing2F
Iterator::Modelod?????!m?~? @)??je?/??1I?<??v@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??x?'??!??????@)|(ђ?Ӣ?1X?r?	@:Preprocessing2U
Iterator::Model::ParallelMapV2,??NG??!9?rA@),??NG??19?rA@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorK?P???!?SlVm???)K?P???1?SlVm???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatet@??$??!2?????)?6?????1pCl\???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??N?0???!?P|???N@)??E?T??1S|%;L??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?:TS?u??!{'
6?x@)?-??č?1??'R??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?4?($???!??]??|??)?4?($???1??]??|??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice??:??T~?!?"?????)??:??T~?1?"?????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeF?=?Ӟr?!?T(&l??)F?=?Ӟr?1?T(&l??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?C??{??!??N????)B???Da?1??bǶ???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorZ???аX?!?t??????)Z???аX?1?t??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?46.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t17.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?R?i@I̜e?O@Q?qڸ[?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/??[<<??/??[<<??!/??[<<??      ??!       "	B]¡? @B]¡? @!B]¡? @*      ??!       2	C??3??C??3??!C??3??:	??3ڪD	@??3ڪD	@!??3ڪD	@B      ??!       J	????2??????2??!????2??R      ??!       Z	????2??????2??!????2??b      ??!       JGPUY?R?i@b q̜e?O@y?qڸ[?>@