	??9̗/*@??9̗/*@!??9̗/*@	?t??n?@?t??n?@!?t??n?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??9̗/*@??捓B??1qX?Q?@A??7/N|??IE??S0@Y??9d/??rEagerKernelExecute 0*	[d;?Oep@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?=yX?5??!7?k??L@)???/J???1???}?J@:Preprocessing2F
Iterator::Model?q?j????!?Y??2b5@)??? ????1#w???+@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatÜ?M???!F???Ĵ.@)?в???1+?5?`)@:Preprocessing2U
Iterator::Model::ParallelMapV2? ??*???!	5??=?@)? ??*???1	5??=?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipm???e??!???As?S@)b?o???1p<?v?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceQ3???U??!?$&l?@)Q3???U??1?$&l?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoran?r?|?!ìBivO@)an?r?|?1ìBivO@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??Hi6???!??? ?M@)"??3?cf?1?%?@???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?34.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s8.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?t??n?@IU
D?1?E@Q'????H@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??捓B????捓B??!??捓B??      ??!       "	qX?Q?@qX?Q?@!qX?Q?@*      ??!       2	??7/N|????7/N|??!??7/N|??:	E??S0@E??S0@!E??S0@B      ??!       J	??9d/????9d/??!??9d/??R      ??!       Z	??9d/????9d/??!??9d/??b      ??!       JGPUY?t??n?@b qU
D?1?E@y'????H@