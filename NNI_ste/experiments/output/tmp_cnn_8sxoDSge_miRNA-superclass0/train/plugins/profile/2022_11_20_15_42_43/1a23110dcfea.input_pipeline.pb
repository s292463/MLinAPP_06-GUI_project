	/??$7 @/??$7 @!/??$7 @	????-@????-@!????-@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL/??$7 @?@?M????1+?6+@Az?I|???I????]???Y??C?.??rEagerKernelExecute 0*	v??/)d@2F
Iterator::Modelm˟o??!?TFFK?G@)X?\T??1????A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatW???????!?Z??cA@)??=?4??1_?R??>@:Preprocessing2U
Iterator::Model::ParallelMapV2yv?և??!F???*@)yv?և??1F???*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceq!??Fʆ?!???@)q!??Fʆ?1???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip&9{ڵ?!r????vJ@)??x?Z???1????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateW#?Ғ?!w?bM?&@)?1>?^?}?1GzL???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??Dׅ|?!M\h-@)??Dׅ|?1M\h-@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapt~??????!??G>*@)?
?.?f?1s??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?23.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9????-@I?/??KD@Q֠N\)?K@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?@?M?????@?M????!?@?M????      ??!       "	+?6+@+?6+@!+?6+@*      ??!       2	z?I|???z?I|???!z?I|???:	????]???????]???!????]???B      ??!       J	??C?.????C?.??!??C?.??R      ??!       Z	??C?.????C?.??!??C?.??b      ??!       JGPUY????-@b q?/??KD@y֠N\)?K@