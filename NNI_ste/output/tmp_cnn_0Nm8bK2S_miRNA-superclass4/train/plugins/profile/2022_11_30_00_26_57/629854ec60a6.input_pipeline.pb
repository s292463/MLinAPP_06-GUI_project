	f???-=*@f???-=*@!f???-=*@      ??!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:f???-=*@-ͭVc??1&????u%@IX;?s? @rEagerKernelExecute 0*	.??臨d@2F
Iterator::Model.??Hٲ?!|??X7F@)]??ky???1?e?????@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??,?"??!^0?'?=@)!Y?nݥ?1p0{?o?9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???o^???!??3@)?Udt@??1???*@:Preprocessing2U
Iterator::Model::ParallelMapV2??N?0???!K?/v)@)??N?0???1K?/v)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????L??!?d<?H@)????L??1?d<?H@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip4??????!?Vp???K@)s??/?x??1???45j@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor\??AA)z?!?l???@)\??AA)z?1?l???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?CQ?O???!?*?Қ5@)?A?d?1Y?H}(??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?15.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI04QX%72@Q????6rT@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	-ͭVc??-ͭVc??!-ͭVc??      ??!       "	&????u%@&????u%@!&????u%@*      ??!       2      ??!       :	X;?s? @X;?s? @!X;?s? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q04QX%72@y????6rT@