	?H?"?@?H?"?@!?H?"?@	B'z%J@B'z%J@!B'z%J@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?H?"?@??_ѭ??1?BB?@A$C??g??I3??bb@Y?{*?=%??rEagerKernelExecute 0*	?????c@2F
Iterator::Model?*???ڳ?!-,]?KH@)?bG?P???1IW=???@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??F;n???!a?	???:@)???i???1?Y$3?6@:Preprocessing2U
Iterator::Model::ParallelMapV2??,????!?S? E-@)??,????1?S? E-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceP?R)v??!اFM	)@)P?R)v??1اFM	)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??Wy??!?Ӣ=??I@)?<???1:tm}??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?wE𿕜?!?-??|1@)-y<-???1??dO?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor0??!?z?!:r?[?N@)0??!?z?1:r?[?N@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%???}???!7???UN3@)\??b??g?138????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?37.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9B'z%J@I?dK&]L@Q6?o*??C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??_ѭ????_ѭ??!??_ѭ??      ??!       "	?BB?@?BB?@!?BB?@*      ??!       2	$C??g??$C??g??!$C??g??:	3??bb@3??bb@!3??bb@B      ??!       J	?{*?=%???{*?=%??!?{*?=%??R      ??!       Z	?{*?=%???{*?=%??!?{*?=%??b      ??!       JGPUYB'z%J@b q?dK&]L@y6?o*??C@