	?S^?@?S^?@!?S^?@	?1!q\@?1!q\@!?1!q\@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?S^?@?D?Ɵ(??1?I?????A?
?rߙ?I?hs??@Y?k????rEagerKernelExecute 0*	??v??Vb@2F
Iterator::ModelG???1??!?? ?ҍI@)	l??3???1IwCCA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat+?~NA??!?k???=@)????=??1j?Y?+I8@:Preprocessing2U
Iterator::Model::ParallelMapV29??ㄙ?!4????0@)9??ㄙ?14????0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceӇ.?o???!b?<X?@)Ӈ.?o???1b?<X?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?Ӹ7?a??!.????x(@) C?*??1??Re??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?5&?\??!?[-rH@)?8?ߡ(??1??h?#?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?T?^??!?E8?^@)?T?^??1?E8?^@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?%?"ܔ?!4vq6`?+@)3?f??c?14`L??d??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?44.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?1!q\@I'??	Q?P@Q ???<@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?D?Ɵ(???D?Ɵ(??!?D?Ɵ(??      ??!       "	?I??????I?????!?I?????*      ??!       2	?
?rߙ??
?rߙ?!?
?rߙ?:	?hs??@?hs??@!?hs??@B      ??!       J	?k?????k????!?k????R      ??!       Z	?k?????k????!?k????b      ??!       JGPUY?1!q\@b q'??	Q?P@y ???<@