	?_?Lyr@?_?Lyr@!?_?Lyr@	?[֎7????[֎7???!?[֎7???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?_?Lyr@<g-mg@1@4???9U@A?@J??ޮ?I?4}v?	7@YxD??????rEagerKernelExecute 0*	??(\??c@2F
Iterator::Model}???E??!?<[?L@)d?g^???1??|'`?C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?M)??Х?!X???:@)??	h"??1??FgW?8@:Preprocessing2U
Iterator::Model::ParallelMapV2S???"???!?_~g`;2@)S???"???1?_~g`;2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat΍?	K<??!?????#@){??????16????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???"[??!|?ä?[E@)'jin??z?1???Dl@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorA??ǘ?v?!?$????@)A??ǘ?v?1?$????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaps??A??!??-N?<@)T?:?g?1????h??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?W?\T[?! ??G????)?W?\T[?1 ??G????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor???M?qZ?!????E??)???M?qZ?1????E??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 63.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?[֎7???I?q??Q@Q??t??<@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	<g-mg@<g-mg@!<g-mg@      ??!       "	@4???9U@@4???9U@!@4???9U@*      ??!       2	?@J??ޮ??@J??ޮ?!?@J??ޮ?:	?4}v?	7@?4}v?	7@!?4}v?	7@B      ??!       J	xD??????xD??????!xD??????R      ??!       Z	xD??????xD??????!xD??????b      ??!       JGPUY?[֎7???b q?q??Q@y??t??<@