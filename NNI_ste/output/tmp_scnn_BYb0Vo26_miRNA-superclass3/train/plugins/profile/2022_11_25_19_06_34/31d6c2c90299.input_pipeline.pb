	?'+???/@?'+???/@!?'+???/@	V?oT??@V?oT??@!V?oT??@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?'+???/@y\T??b??1????s? @I9??!;)@Y??ȑ΀??r0*	     .v@2U
Iterator::Model::ParallelMapV2qs* ???!*??$&K@)qs* ???1*??$&K@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatcB?%Uۭ?!?fK??n0@))??????1?~'Fzl+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???4????!I???B@)i??TN??1V?f?8@%@:Preprocessing2F
Iterator::Model1???z??!? ??nYO@)?ګ????11??s? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??I?????!?#J?|$@)6Φ#????1e???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??ّ?;??!????0@)??ّ?;??1????0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorbe4?yŃ?!x<?1^?@)be4?yŃ?1x<?1^?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?79.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9V?oT??@I?26bU?S@QO?D??*@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	y\T??b??y\T??b??!y\T??b??      ??!       "	????s? @????s? @!????s? @*      ??!       2      ??!       :	9??!;)@9??!;)@!9??!;)@B      ??!       J	??ȑ΀????ȑ΀??!??ȑ΀??R      ??!       Z	??ȑ΀????ȑ΀??!??ȑ΀??b      ??!       JGPUYV?oT??@b q?26bU?S@yO?D??*@