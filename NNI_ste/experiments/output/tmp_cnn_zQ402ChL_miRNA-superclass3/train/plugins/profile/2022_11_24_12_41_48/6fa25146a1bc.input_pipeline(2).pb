	n?@W?o@n?@W?o@!n?@W?o@	w?Q????w?Q????!w?Q????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLn?@W?o@|?/$?c@1|~!iS@A??9?ا?I??TN31@Y?~7ݲ??rEagerKernelExecute 0*	??Q?e@2F
Iterator::ModelhZbe4???!q???qI@)?ΤMխ?1'?XKA@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?f?v???!G]?Ēe?@)c?=yX??1?d3??9<@:Preprocessing2U
Iterator::Model::ParallelMapV2?g?K6??!?????L0@)?g?K6??1?????L0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???????!m{eå~%@)}?;l"3??1!????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??Li?-??!??A?H@)\t??z???1?86???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??3?{?!?F??v@)??3?{?1?F??v@:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensorW??x??i?!?ϡ????)W??x??i?1?ϡ????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapv3????!??+? ?@@)???Z(i?1??$??*??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceL??1%b?!e?Os?	??)L??1%b?1e?Os?	??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 62.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9x?Q????I?U?SQ@Q?7\?>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	|?/$?c@|?/$?c@!|?/$?c@      ??!       "	|~!iS@|~!iS@!|~!iS@*      ??!       2	??9?ا???9?ا?!??9?ا?:	??TN31@??TN31@!??TN31@B      ??!       J	?~7ݲ???~7ݲ??!?~7ݲ??R      ??!       Z	?~7ݲ???~7ݲ??!?~7ݲ??b      ??!       JGPUYx?Q????b q?U?SQ@y?7\?>@