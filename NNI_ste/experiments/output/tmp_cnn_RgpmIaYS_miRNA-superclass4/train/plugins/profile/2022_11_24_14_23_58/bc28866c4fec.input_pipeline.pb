	??B?5@??B?5@!??B?5@	=(\??D @=(\??D @!=(\??D @"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??B?5@+???,?@1???P? +@A9?? n??I???;????Y6Φ#????rEagerKernelExecute 0*	|?5^?g@2F
Iterator::Model???h????!B?g?OwL@) ???WW??1??PB@:Preprocessing2U
Iterator::Model::ParallelMapV2)?A&9??!Ex?0iM4@))?A&9??1Ex?0iM4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatyY|??!ds?ܰ6@)!???3??1K???`93@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??yȔ??!?????"@)??yȔ??1?????"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip28J^?c??!?<?K??E@)x?ܙ	???1䷹???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??	????!1?Pɮ
+@)H4?"??1??y?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor@?Z?kBz?!]????@)@?Z?kBz?1]????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapޓ??ZӜ?!?N???q.@))w????i?1J?d?8??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 29.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9=(\??D @I?H?!B@Q?\)?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	+???,?@+???,?@!+???,?@      ??!       "	???P? +@???P? +@!???P? +@*      ??!       2	9?? n??9?? n??!9?? n??:	???;???????;????!???;????B      ??!       J	6Φ#????6Φ#????!6Φ#????R      ??!       Z	6Φ#????6Φ#????!6Φ#????b      ??!       JGPUY=(\??D @b q?H?!B@y?\)?N@