	?!??@?!??@!?!??@	?[??#@?[??#@!?[??#@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?!??@?? v????1??릔W@A???n??I3?`??I@Y9?ߡ(???rEagerKernelExecute 0*	????ؐ@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??Ά?3??!C?O???P@)#?#?)??1D??uP@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapq:???!U`*?9@)L⬈????18?,V??5@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat_'?ei???!A;?(j@)|?o?^}??15~A)?@:Preprocessing2F
Iterator::Model?Ù_???!?4???@)Jy???16.zW?@:Preprocessing2U
Iterator::Model::ParallelMapV2??ص?ݒ?!.v}?V??)??ص?ݒ?1.v}?V??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?
?|$%??!&?	??)??}q?J??1ו`?????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat5???{??!$&?+???)?f??I}??1?Z??w??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch? ݗ3ۅ?!j?????)? ݗ3ۅ?1j?????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip#gaO;???!??:??`Q@)???ދ/z?1??J}???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?0DN_?w?!??^u@??)?0DN_?w?1??^u@??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???D?v?!Ԗ?m???)???D?v?1Ԗ?m???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?)?TPq?!b?g????)?)?TPq?1b?g????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate&4I,)w?!?\D?p???)?x'?^?1q??/??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensord?????M?!??W??|??)d?????M?1??W??|??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 14.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?43.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?[??#@I????,=M@Q_?Ǘ?C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?? v?????? v????!?? v????      ??!       "	??릔W@??릔W@!??릔W@*      ??!       2	???n?????n??!???n??:	3?`??I@3?`??I@!3?`??I@B      ??!       J	9?ߡ(???9?ߡ(???!9?ߡ(???R      ??!       Z	9?ߡ(???9?ߡ(???!9?ߡ(???b      ??!       JGPUY?[??#@b q????,=M@y_?Ǘ?C@