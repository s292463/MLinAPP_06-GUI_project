	:??KT?@:??KT?@!:??KT?@	?g?s?m???g?s?m??!?g?s?m??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL:??KT?@v???_?@13???/??A?p?;??I???jdW@Y??~?7??rEagerKernelExecute 0*	?"?????@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?
?.H??!????H@)㪲?????1H;O??F@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?k?????!R	-P??D@)??c?g^??1??sPC@:Preprocessing2F
Iterator::ModelxB???ϱ?!??+?@@)6?e?Ԩ?1??&???@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??1z??!?w?C<?@)Ӿ??z??1?fX??@:Preprocessing2U
Iterator::Model::ParallelMapV2u/3l???!\X?Y^b??)u/3l???1\X?Y^b??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??ek}???!?Kt???)p\?M4??1???G~>??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??
???!????$???)TpxADj??1c?ʖ????:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetchx?W?L??!U?l?6???)x?W?L??1U?l?6???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?7?06??!?*??"F@)2t??ׁ?1r?R?J??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????z?!?*??f???)????z?1?*??f???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice\>???v?!?{z???)\>???v?1?{z???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeZ?rL?o?!&?{\G???)Z?rL?o?1&?{\G???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?X?? ~?!t#?	Rm??)j???]?1?'	T^??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?mO???N?!???7`??)?mO???N?1???7`??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 30.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?41.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?g?s?m??I@?	2?Q@Q?x?A_`:@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	v???_?@v???_?@!v???_?@      ??!       "	3???/??3???/??!3???/??*      ??!       2	?p?;???p?;??!?p?;??:	???jdW@???jdW@!???jdW@B      ??!       J	??~?7????~?7??!??~?7??R      ??!       Z	??~?7????~?7??!??~?7??b      ??!       JGPUY?g?s?m??b q@?	2?Q@y?x?A_`:@