	??t ?@??t ?@!??t ?@	??9??@??9??@!??9??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??t ?@n?8)?{??1od??A??A?\?????I?|zl? 
@Y??Wy??rEagerKernelExecute 0*	??S??a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatܝ??.4??!b??R?/@@)??'+????1??Ǝq;@:Preprocessing2F
Iterator::Modelw?ӂ}??!6@~
>?D@)?}iƢ?1orV??1:@:Preprocessing2U
Iterator::Model::ParallelMapV2?٬?\m??!?L9,?-@)?٬?\m??1?L9,?-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicep	???J??!?Q?O,@)p	???J??1?Q?O,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???????!]%? &V3@)5&?\R?}?18???w?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??b????!ɿ???mM@)?2??(}?1?L??rW@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?n?EE|?!`?x??@)*?n?EE|?1`?x??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?h????!?[???e5@)?^?sa?g?1ر)0#~ @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?56.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t12.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??9??@IW?;l?ZQ@QyQBo?8@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	n?8)?{??n?8)?{??!n?8)?{??      ??!       "	od??A??od??A??!od??A??*      ??!       2	?\??????\?????!?\?????:	?|zl? 
@?|zl? 
@!?|zl? 
@B      ??!       J	??Wy????Wy??!??Wy??R      ??!       Z	??Wy????Wy??!??Wy??b      ??!       JGPUY??9??@b qW?;l?ZQ@yyQBo?8@