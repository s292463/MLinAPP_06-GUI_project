	K?*n?@K?*n?@!K?*n?@	??4?@??4?@!??4?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCK?*n?@L????q??1?'eR?@I????x?@YT???
???rEagerKernelExecute 0*	J+??c@2F
Iterator::Model???%ǝ??!?~???G@)9c??ɩ?1?>?,???@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatݔ?Z	ݥ?!?a?4J;@)m???L??1,??ĭ6@:Preprocessing2U
Iterator::Model::ParallelMapV2-??o????!}?(?^,@)-??o????1}?(?^,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?q??Q???!c?O?J@)N)???]??1P?L??&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?! 8????!Ҥ3? @)?! 8????1Ҥ3? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?? ??ԕ?!??g?7+@)?W?ۼ??1)?g?3?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?[w?T?|?!,???@)?[w?T?|?1,???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2s??c͘?!#???̽.@)?????g?1ۙ?ۨt??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?29.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??4?@I`S?[?@@Q@?? ?O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	L????q??L????q??!L????q??      ??!       "	?'eR?@?'eR?@!?'eR?@*      ??!       2      ??!       :	????x?@????x?@!????x?@B      ??!       J	T???
???T???
???!T???
???R      ??!       Z	T???
???T???
???!T???
???b      ??!       JGPUY??4?@b q`S?[?@@y@?? ?O@