	WZF?=?@WZF?=?@!WZF?=?@	??RU}?????RU}???!??RU}???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLWZF?=?@?c> ?y@1?б?J\??AmU???I??>@Yg?!?{??rEagerKernelExecute 0*	ObX9.v@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?C3O?)??!olP??J@)?9?m½??1???-?I@:Preprocessing2F
Iterator::Model?3?%??!??W??G8@)1y?|??1zf???1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????x!??!?X???2@)"R?.????1ot??;?0@:Preprocessing2U
Iterator::Model::ParallelMapV2????r-??!cV?B?@)????r-??1cV?B?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice]?P????!iBg-		@)]?P????1iBg-		@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip毐?2??!0 *?R@)T1??c??1?1Hmq@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensork?#?]J}?!?!Y? @)k?#?]J}?1?!Y? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????y???!??q?K@)K????2i?1rzTAj???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 28.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?44.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??RU}???I?O?ָUR@Q?E?t?8@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?c> ?y@?c> ?y@!?c> ?y@      ??!       "	?б?J\???б?J\??!?б?J\??*      ??!       2	mU???mU???!mU???:	??>@??>@!??>@B      ??!       J	g?!?{??g?!?{??!g?!?{??R      ??!       Z	g?!?{??g?!?{??!g?!?{??b      ??!       JGPUY??RU}???b q?O?ָUR@y?E?t?8@