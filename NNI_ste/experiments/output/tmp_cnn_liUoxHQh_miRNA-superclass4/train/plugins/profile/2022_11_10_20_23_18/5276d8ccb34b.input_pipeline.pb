	?1 ?"@?1 ?"@!?1 ?"@	???0]j?????0]j??!???0]j??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?1 ?"@??P?l@1?+?S?@A?k*??I?wcAa?@Y?؀q??rEagerKernelExecute 0*	??ʡEa@2F
Iterator::Model?????j??!??w??F@)??ٮ???16??rJ;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?w??,??!A9A??@)K?b??¢?1?e(4?:@:Preprocessing2U
Iterator::Model::ParallelMapV2?n??Ř?!????1@)?n??Ř?1????1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????!?JJgV#@)?????1?JJgV#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate7??t??!?Ό?~0@)????@??1E???N-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?>??s(??!?+-?@yK@)? 3??O|?1<?N ?L@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?LN?S{?!7l?d??@)?LN?S{?17l?d??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?J[\?3??!K\.2@)??f??e?1??h????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 24.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?49.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???0]j??I????x?R@Q???vw8@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??P?l@??P?l@!??P?l@      ??!       "	?+?S?@?+?S?@!?+?S?@*      ??!       2	?k*???k*??!?k*??:	?wcAa?@?wcAa?@!?wcAa?@B      ??!       J	?؀q???؀q??!?؀q??R      ??!       Z	?؀q???؀q??!?؀q??b      ??!       JGPUY???0]j??b q????x?R@y???vw8@