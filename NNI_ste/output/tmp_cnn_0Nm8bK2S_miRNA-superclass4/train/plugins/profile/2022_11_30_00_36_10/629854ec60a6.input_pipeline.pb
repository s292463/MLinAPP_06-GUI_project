	Qi??>?:@Qi??>?:@!Qi??>?:@	??L֜[????L֜[??!??L֜[??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLQi??>?:@?m??*@1????%@Aǜg?K6??I?2?}?E??Y???tw???rEagerKernelExecute 0*	7?A`??c@2F
Iterator::Model?5??Wt??!????iAH@))?7Ӆ??1????>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatL<???!?vv??M?@)?z????1{?9"?r;@:Preprocessing2U
Iterator::Model::ParallelMapV2 UܸŜ?!1?????1@) UܸŜ?11?????1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?"j??G??!?Z???@)?"j??G??1?Z???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?\QJV??!d?????*@) ?8?@d??1Ŵ????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?? w???!!	;??I@)?u??$???1P??S @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorm:?Y?x?!?q?9?@)m:?Y?x?1?q?9?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?c#????!nki???-@)D6?.6?d?1MX?>~???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??L֜[??I??6x?M@Q5???TD@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?m??*@?m??*@!?m??*@      ??!       "	????%@????%@!????%@*      ??!       2	ǜg?K6??ǜg?K6??!ǜg?K6??:	?2?}?E???2?}?E??!?2?}?E??B      ??!       J	???tw??????tw???!???tw???R      ??!       Z	???tw??????tw???!???tw???b      ??!       JGPUY??L֜[??b q??6x?M@y5???TD@