	j4?K"@j4?K"@!j4?K"@	y[6!?@y[6!?@!y[6!?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLj4?K"@-??m??1?V횐f@A;?O??nr?II?5C???Y{??????rEagerKernelExecute 0*	z?G?w@2F
Iterator::Model	oB@???! x?9?R@)e?F ^???1?㺖?P@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????>9??!?i???+@)?ۻ}???1?>?O)'@:Preprocessing2U
Iterator::Model::ParallelMapV2g??)??!8??;??@)g??)??18??;??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;???!?)N??@);???1?)N??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?7?k????!C??ȸ6@)'??b??1?? ??]@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipW?'???!?V?8@)?????m??1?탁?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?|???!~>? ?:@)y?|???1~>? ?:@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??AB?/??!?????!@)A??ǘ?f?1??5?e??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?15.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9y[6!?@ID ???"2@Q7?;? hS@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	-??m??-??m??!-??m??      ??!       "	?V횐f@?V횐f@!?V횐f@*      ??!       2	;?O??nr?;?O??nr?!;?O??nr?:	I?5C???I?5C???!I?5C???B      ??!       J	{??????{??????!{??????R      ??!       Z	{??????{??????!{??????b      ??!       JGPUYy[6!?@b qD ???"2@y7?;? hS@