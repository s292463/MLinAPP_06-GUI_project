	2t??@2t??@!2t??@	?((??;@?((??;@!?((??;@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL2t??@??c???1??AϦ
@A???????I	kc???@Y????`??rEagerKernelExecute 0*	$??CWc@2F
Iterator::Model??k*??!???d1H@)Va3?٪?1??????@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??????!?<&??:@)k`??á?1Dsr?al6@:Preprocessing2U
Iterator::Model::ParallelMapV27???????!????,@)7???????1????,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??9??q??!_JX???I@)?ĬC9??17P/Խ%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??8G??!???[2-@)??8G??1???[2-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate-$`ty??!???(?(@)?l????1????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?{??Pkz?!mJ&???@)?{??Pkz?1mJ&???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap"???k??!??1?M,@)??|??g?1?;?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?36.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?((??;@I؂?}_J@Q???O?LF@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??c?????c???!??c???      ??!       "	??AϦ
@??AϦ
@!??AϦ
@*      ??!       2	??????????????!???????:		kc???@	kc???@!	kc???@B      ??!       J	????`??????`??!????`??R      ??!       Z	????`??????`??!????`??b      ??!       JGPUY?((??;@b q؂?}_J@y???O?LF@