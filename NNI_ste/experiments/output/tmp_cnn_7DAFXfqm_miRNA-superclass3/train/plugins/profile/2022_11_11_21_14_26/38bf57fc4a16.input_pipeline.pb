	
If?G@
If?G@!
If?G@	?*?	??@?*?	??@!?*?	??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL
If?G@?T1????1|{נ/}@A?磌? ??IZ??ڊ?@Y??JY?8??rEagerKernelExecute 0*	??(\?Hp@2U
Iterator::Model::ParallelMapV2}A	]??!???nɈH@)}A	]??1???nɈH@:Preprocessing2F
Iterator::Modelĳ??!"!?ӍP@)????ߦ?1Y5???%1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat$}ZE??!??? Q]2@)?5?eܤ?1?>?F/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???C?r??!?t??G @)???C?r??1?t??G @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?R?{/??!߄q??&@)?z??9y??1??aN?2
@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip;7m?i???!??KX?@@)??Cl??1?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorCqǛ?}?!?Ue
??@)CqǛ?}?1?Ue
??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??4}v??!l`?ܮ(@)??}???e?1?g܆o??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?41.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?*?	??@I?A?i??L@Q?K??۱C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?T1?????T1????!?T1????      ??!       "	|{נ/}@|{נ/}@!|{נ/}@*      ??!       2	?磌? ???磌? ??!?磌? ??:	Z??ڊ?@Z??ڊ?@!Z??ڊ?@B      ??!       J	??JY?8????JY?8??!??JY?8??R      ??!       Z	??JY?8????JY?8??!??JY?8??b      ??!       JGPUY?*?	??@b q?A?i??L@y?K??۱C@