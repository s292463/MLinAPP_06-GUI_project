	<?H???-@<?H???-@!<?H???-@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC<?H???-@?6?Nx	??1?ZH??%@A>ʈ@???IpxADj? @rEagerKernelExecute 0*	??Q??t@2F
Iterator::Model,cC7???!?y??B?R@)DkE?????1f?????P@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????A_??!w??/@)V???̯??1~.K{?*@:Preprocessing2U
Iterator::Model::ParallelMapV2??@????!?xn??[@)??@????1?xn??[@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????G6??!P????P@)????G6??1P????P@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateo?[t???!??yw[@)?);??.??1S?W?te@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipv?Kp???!-z??8@) :̗`?1?lu??u@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorմ?i?{}?!?M???X@)մ?i?{}?1?M???X@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapx	N} y??!i???e?@)???
a5f?1?mPk"??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?14.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noId?{W],:@Q'!??tR@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?6?Nx	???6?Nx	??!?6?Nx	??      ??!       "	?ZH??%@?ZH??%@!?ZH??%@*      ??!       2	>ʈ@???>ʈ@???!>ʈ@???:	pxADj? @pxADj? @!pxADj? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qd?{W],:@y'!??tR@