	LnYk.@LnYk.@!LnYk.@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCLnYk.@3??????1y?n?|J'@A????Fu??I9??cx???rEagerKernelExecute 0*	?A`??d@2F
Iterator::ModelJ^?c@???!B??S?K@)eq???б?1<?E`"?E@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat()? ???!?k?l9@)?}9?]??1??]5@:Preprocessing2U
Iterator::Model::ParallelMapV2??wF[???!L ??)@)??wF[???1L ??)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?ZC?????!??Q?"h@)?ZC?????1??Q?"h@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?{?i????!??Yl?G(@)М?)?d??1??a?'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???,&??!??q|?F@)1???z??1???9@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor@?Z?kBz?!?w?@)@?Z?kBz?1?w?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap]?jJ???!		M?,@)?\p?h?1?:?Gq??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 10.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?12.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI? ]o7@Q???>$S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	3??????3??????!3??????      ??!       "	y?n?|J'@y?n?|J'@!y?n?|J'@*      ??!       2	????Fu??????Fu??!????Fu??:	9??cx???9??cx???!9??cx???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q? ]o7@y???>$S@