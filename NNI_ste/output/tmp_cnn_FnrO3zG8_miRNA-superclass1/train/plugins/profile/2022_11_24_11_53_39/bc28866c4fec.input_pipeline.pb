	?5Φ# @?5Φ# @!?5Φ# @	NM*?@NM*?@!NM*?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?5Φ# @?\???J??1qVDM??@A?! _B??I&?(??@Y?/?x????rEagerKernelExecute 0*	?l???_?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapDio??I??!?y?xK@)!??	L???1?????H@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?f?lt???!?A??KA@)????y??12??H?=@:Preprocessing2F
Iterator::Model~?k?,	??!xM??"? @)!XU/?Ӭ?1.?b??@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?u??ť??!????%@)?e?ikD??1?RVvz+@:Preprocessing2U
Iterator::Model::ParallelMapV2???4}??!??J ?@)???4}??1??J ?@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch???V_]??!?J?A??@)???V_]??1?J?A??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatec?#?w~??!,?B???)%??1??1?G,????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?։??
??!I'??? @)1?Zd??1:tX???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?ri??+??!	??u?L@)#?ng_y??1Ψ?ےb??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorVJ??cy?!)????)VJ??cy?1)????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?\?mO?x?!4?}#?j??)?\?mO?x?14?}#?j??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangep}Xo?
s?!??\x???)p}Xo?
s?1??\x???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate|?????!????.??){?\?&?[?1g~?y??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?t?_??T?!dU??G??)?t?_??T?1dU??G??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?40.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9NM*?@IP? ??M@Q'??yB@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?\???J???\???J??!?\???J??      ??!       "	qVDM??@qVDM??@!qVDM??@*      ??!       2	?! _B???! _B??!?! _B??:	&?(??@&?(??@!&?(??@B      ??!       J	?/?x?????/?x????!?/?x????R      ??!       Z	?/?x?????/?x????!?/?x????b      ??!       JGPUYNM*?@b qP? ??M@y'??yB@