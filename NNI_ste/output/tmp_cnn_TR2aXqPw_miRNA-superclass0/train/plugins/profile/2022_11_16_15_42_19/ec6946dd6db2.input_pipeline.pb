	????g@????g@!????g@	.Yߖ@.Yߖ@!.Yߖ@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL????g@%!??q??1C?Y?????A??ĭ???I(Hlw?@Y?`?d7??rEagerKernelExecute 0*	??x?&?r@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate$Di???!??QO?I@)ٴR???1:??E<F@:Preprocessing2F
Iterator::ModelC8???!?L|K??<@)*Ŏơ~??1P?Un	?4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?t"?T??!???I?,)@)e?,?i???1I?S??$@:Preprocessing2U
Iterator::Model::ParallelMapV2??6S!??!D?L?[ @)??6S!??1D?L?[ @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceVҊo(|??!??YRH@)VҊo(|??1??YRH@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipp??1=a??!?? ??Q@)??h?????1@???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?f?ba?|?!f???K?@)?f?ba?|?1f???K?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap.rOWw,??!D????EJ@)???B??b?1ԙ??e??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?54.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t21.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9,Yߖ@I?? ?S@QH?8?_[0@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	%!??q??%!??q??!%!??q??      ??!       "	C?Y?????C?Y?????!C?Y?????*      ??!       2	??ĭ?????ĭ???!??ĭ???:	(Hlw?@(Hlw?@!(Hlw?@B      ??!       J	?`?d7???`?d7??!?`?d7??R      ??!       Z	?`?d7???`?d7??!?`?d7??b      ??!       JGPUY,Yߖ@b q?? ?S@yH?8?_[0@