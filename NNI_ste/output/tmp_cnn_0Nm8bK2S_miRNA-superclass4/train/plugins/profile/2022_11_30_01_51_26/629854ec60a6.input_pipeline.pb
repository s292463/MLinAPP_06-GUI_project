	wJ?[5@wJ?[5@!wJ?[5@	??U???????U?????!??U?????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCwJ?[5@<Nё\~??1?c!:n'@I?w?Gop!@Y??s?/??rEagerKernelExecute 0*	R???	f@2F
Iterator::Model??A%?c??!Z???K@)?o
+T??1????B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??Li?-??!?nsN??;@) ~?{?ڥ?1P???58@:Preprocessing2U
Iterator::Model::ParallelMapV2F?n?1??!S??1@)F?n?1??1S??1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice^f?(???!??%f@)^f?(???1??%f@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???p?Q??!u?țɝ'@)+??????1??k?w?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?V??????!?q3?a?F@)~oӟ?H??1??Q?%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?drjg?z?!_D??4v@)?drjg?z?1_D??4v@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapAJ?i???!	????*@)R?=?Ne?1??Qٗ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?40.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??U?????I??????E@Q???vlK@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	<Nё\~??<Nё\~??!<Nё\~??      ??!       "	?c!:n'@?c!:n'@!?c!:n'@*      ??!       2      ??!       :	?w?Gop!@?w?Gop!@!?w?Gop!@B      ??!       J	??s?/????s?/??!??s?/??R      ??!       Z	??s?/????s?/??!??s?/??b      ??!       JGPUY??U?????b q??????E@y???vlK@