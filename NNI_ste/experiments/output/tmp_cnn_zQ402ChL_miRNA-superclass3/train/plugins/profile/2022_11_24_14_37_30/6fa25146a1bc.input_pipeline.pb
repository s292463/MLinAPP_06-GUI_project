	??C? @??C? @!??C? @	??ț??
@??ț??
@!??ț??
@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??C? @?9D?\??19?)9'6??Az??{??I?4?8E'@Y?f????rEagerKernelExecute 0*	??????c@2F
Iterator::Model?=yX???!????H?J@)b?qm???1?b??+C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?}??g??!l??9'9@)??D????1??x?&?4@:Preprocessing2U
Iterator::Model::ParallelMapV29
p??!O#ܸ .@)9
p??1O#ܸ .@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateo?????!?޵??~-@)AJ?i??1s%?+?? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?,??o??!?r)~2@)?,??o??1?r)~2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip.???1???!3?LG@)I?V??1)????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?zNz??z?!N?REL?@)?zNz??z?1N?REL?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap}?E֚?!K:????0@)b???LLg?1[?$t???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 22.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?53.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??ț??
@ID????$S@Q???,O4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?9D?\???9D?\??!?9D?\??      ??!       "	9?)9'6??9?)9'6??!9?)9'6??*      ??!       2	z??{??z??{??!z??{??:	?4?8E'@?4?8E'@!?4?8E'@B      ??!       J	?f?????f????!?f????R      ??!       Z	?f?????f????!?f????b      ??!       JGPUY??ț??
@b qD????$S@y???,O4@