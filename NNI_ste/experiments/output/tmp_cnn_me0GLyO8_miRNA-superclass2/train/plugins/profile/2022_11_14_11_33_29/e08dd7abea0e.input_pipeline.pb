	7R?H??$@7R?H??$@!7R?H??$@	?_d-?????_d-????!?_d-????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL7R?H??$@A?m%@1]??? @A??\??k??I??:/@Y?n?????rEagerKernelExecute 0*	l?????b@2F
Iterator::Model,??d??!2?AG&0E@)A	]ޤ?1?˔q??:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??_vO??!گy???<@)?e?آ?1?Q??\8@:Preprocessing2U
Iterator::Model::ParallelMapV2-^,?ӗ?!e?9??.@)-^,?ӗ?1e?9??.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceW??m??!?T??p?+@)W??m??1?T??p?+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate? ??ǟ?!"O??[?4@)???r???1Z??Y??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipDQ?O?I??!?@????L@)????#*??1<???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????r-z?!ap??c?@)????r-z?1ap??c?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???8?j??!3?????6@))??qh?1?#k???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?57.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?_d-????IV??ήS@Q?aH???3@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	A?m%@A?m%@!A?m%@      ??!       "	]??? @]??? @!]??? @*      ??!       2	??\??k????\??k??!??\??k??:	??:/@??:/@!??:/@B      ??!       J	?n??????n?????!?n?????R      ??!       Z	?n??????n?????!?n?????b      ??!       JGPUY?_d-????b qV??ήS@y?aH???3@