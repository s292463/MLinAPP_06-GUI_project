	bHN&nu@bHN&nu@!bHN&nu@	[F??k?!@[F??k?!@![F??k?!@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLbHN&nu@??:?J??1??8?:??A??b?d??IK\Ǹ??@Y????c???rEagerKernelExecute 0*	X9??v?f@2F
Iterator::Model?+??f*??!|-踎E@)??6?ُ??1]H?&??>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;oc?#կ?!??=?A@)7P??|z??1??E?q>@:Preprocessing2U
Iterator::Model::ParallelMapV2f??t牗?!9%S?))@)f??t牗?19%S?))@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipz?(???!???GqL@)[`???f??1Suzb?$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice46<?R??!?~#?.?@)46<?R??1?~#?.?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate@j'?;??!??k??%@)L??1%??1k?,?e@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??8?z?!>y?Y?@)??8?z?1>y?Y?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??K?[??!?Ӊ???(@)?'i?h?1E8????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?44.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t18.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9[F??k?!@I]~3?O@Q?????;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??:?J????:?J??!??:?J??      ??!       "	??8?:????8?:??!??8?:??*      ??!       2	??b?d????b?d??!??b?d??:	K\Ǹ??@K\Ǹ??@!K\Ǹ??@B      ??!       J	????c???????c???!????c???R      ??!       Z	????c???????c???!????c???b      ??!       JGPUY[F??k?!@b q]~3?O@y?????;@