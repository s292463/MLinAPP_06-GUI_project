	_??x??@_??x??@!_??x??@	??p???@??p???@!??p???@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL_??x??@b?k_@??1????O@A1A?º??Ia?????@Y!˂?????rEagerKernelExecute 0*	i?t?d@2F
Iterator::Modelo??Ia޳?!?7nc?(H@)c???J??1?ɥr??A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatu?V??!E??CF?9@)?O ?Ȓ??1???5^5@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?I?O?c??!h&??q9+@)?I?O?c??1h&??q9+@:Preprocessing2U
Iterator::Model::ParallelMapV2?????n??!??!û*@)?????n??1??!û*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?s]?@??!tȑ?P?I@)?#???E??1ƾWl?7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??*?C3??!??u???2@)??T????1?*' m@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??i?{?!u??B?@)??i?{?1u??B?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap<K?P???!??5?j?4@){?G?zd?1??l????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?47.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??p???@IHX?? ?H@Q??x@?G@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	b?k_@??b?k_@??!b?k_@??      ??!       "	????O@????O@!????O@*      ??!       2	1A?º??1A?º??!1A?º??:	a?????@a?????@!a?????@B      ??!       J	!˂?????!˂?????!!˂?????R      ??!       Z	!˂?????!˂?????!!˂?????b      ??!       JGPUY??p???@b qHX?? ?H@y??x@?G@