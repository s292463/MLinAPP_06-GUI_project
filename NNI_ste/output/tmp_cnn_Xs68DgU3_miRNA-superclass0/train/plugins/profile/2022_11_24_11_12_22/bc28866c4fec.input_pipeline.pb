	???8?$@???8?$@!???8?$@	?????? @?????? @!?????? @"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???8?$@I*S?A???1f?c]?v@AJ?>?ɛ?I"7??< @Y?V?/?'??rEagerKernelExecute 0*	X9??vu@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???r-Z??!Z(Օ?Q@)g??67???1?N??A0Q@:Preprocessing2F
Iterator::Model?Uס????!W?Y?73@)??b??1[?a??)@:Preprocessing2U
Iterator::Model::ParallelMapV2?g?RD???!??#???@)?g?RD???1??#???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipG:#/k??!ꀩ92T@)ܹ0ҋڍ?1??i?N@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatxF[?D???!r???{?@)????p???1?	??%K
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor^H??0~z?!E?J?????)^H??0~z?1E?J?????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor???c[l?!]???+???)???c[l?1]???+???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap" 8?????!7?G??Q@).????g?1W?ؚ????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????3?`?!??&????)????3?`?1??&????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 17.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?20.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?????? @I?Z?R??B@Q?:^D5N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	I*S?A???I*S?A???!I*S?A???      ??!       "	f?c]?v@f?c]?v@!f?c]?v@*      ??!       2	J?>?ɛ?J?>?ɛ?!J?>?ɛ?:	"7??< @"7??< @!"7??< @B      ??!       J	?V?/?'???V?/?'??!?V?/?'??R      ??!       Z	?V?/?'???V?/?'??!?V?/?'??b      ??!       JGPUY?????? @b q?Z?R??B@y?:^D5N@