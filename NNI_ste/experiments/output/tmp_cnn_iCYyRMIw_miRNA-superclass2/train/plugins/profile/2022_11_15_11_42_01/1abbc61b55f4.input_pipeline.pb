	w?
??n&@w?
??n&@!w?
??n&@	dP:i??dP:i??!dP:i??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLw?
??n&@??c> ???1????.??A???????I?????@Y[??g͏??rEagerKernelExecute 0*	4333??@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate;??"?@!M??-?bX@)Nc{-??@1?b4?XX@:Preprocessing2F
Iterator::ModelB?Ѫ?t??!sZ?O?@??)G??????1?#??<P??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat/?.?H??!??_V???)??×???1?F?????:Preprocessing2U
Iterator::Model::ParallelMapV2x?g?ɗ?!?!?_c??)x?g?ɗ?1?!?_c??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??tw?=@!??&??X@)/??|?X??1??X???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??K?u??!^`|??A??)??K?u??1^`|??A??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorJ	??z?}?!?k=?6??)J	??z?}?1?k=?6??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap]m????@!?o.??eX@)?!??l?1??*ubӪ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 13.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?70.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9dP:i??IA?H;B?T@Q&?o? --@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??c> ?????c> ???!??c> ???      ??!       "	????.??????.??!????.??*      ??!       2	??????????????!???????:	?????@?????@!?????@B      ??!       J	[??g͏??[??g͏??![??g͏??R      ??!       Z	[??g͏??[??g͏??![??g͏??b      ??!       JGPUYdP:i??b qA?H;B?T@y&?o? --@