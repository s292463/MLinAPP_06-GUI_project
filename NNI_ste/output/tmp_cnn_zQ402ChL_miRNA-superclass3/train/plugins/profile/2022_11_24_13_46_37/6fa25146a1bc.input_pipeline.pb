	d????\!@d????\!@!d????\!@	o???Y	@o???Y	@!o???Y	@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLd????\!@??đ??1????( ??A1a4+ۇ??I??:TS?@Y0c
?8???rEagerKernelExecute 0*	?V?c@2F
Iterator::ModelGv?e?޳?! ????H@),?????1Of??&3?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat? ??F!??!??5??g?@)C?=?Х?1eNʰkC;@:Preprocessing2U
Iterator::Model::ParallelMapV2?j,am???!????v2@)?j,am???1????v2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice4,F]k???!??g^??@)4,F]k???1??g^??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?????!??!??_???'@)I??r?S??1?IW??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip-AF@?#??!?A?4
+I@)?<???1??*4Y@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??c${?z?!?l????@)??c${?z?1?l????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????????!?Q???/+@)??g??d?1??e8??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?56.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9p???Y	@I?m?s,?S@Q??#R?2@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??đ????đ??!??đ??      ??!       "	????( ??????( ??!????( ??*      ??!       2	1a4+ۇ??1a4+ۇ??!1a4+ۇ??:	??:TS?@??:TS?@!??:TS?@B      ??!       J	0c
?8???0c
?8???!0c
?8???R      ??!       Z	0c
?8???0c
?8???!0c
?8???b      ??!       JGPUYp???Y	@b q?m?s,?S@y??#R?2@