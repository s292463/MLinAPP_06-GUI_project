	?
(?ӿ%@?
(?ӿ%@!?
(?ӿ%@	??F???@??F???@!??F???@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?
(?ӿ%@?p?a????1??V?@AjK??`??I?7M?p??Yz?蹅???rEagerKernelExecute 0*	/?$?e@2F
Iterator::Model??1????!A脅?aF@)3???yS??1?+CbEl<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate3?`?????!???ӏ?;@)?kzPP???1??=?K9@:Preprocessing2U
Iterator::Model::ParallelMapV2:̗`??!??ƨ?W0@):̗`??1??ƨ?W0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip;9CqǛ??!?{z?K@)??U????1?A?u?]/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?<c_????!>rRe#@)^K?=???1?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?\?mO?x?!???s=?@)?\?mO?x?1???s=?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??;Ū?!??Q=>@)RH2?w?m?1}??i? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??$?pte?!?hv??)??$?pte?1?hv??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?>?`?!\???{???)?>?`?1\???{???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?13.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??F???@IT?s5k?9@Q??m??Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?p?a?????p?a????!?p?a????      ??!       "	??V?@??V?@!??V?@*      ??!       2	jK??`??jK??`??!jK??`??:	?7M?p???7M?p??!?7M?p??B      ??!       J	z?蹅???z?蹅???!z?蹅???R      ??!       Z	z?蹅???z?蹅???!z?蹅???b      ??!       JGPUY??F???@b qT?s5k?9@y??m??Q@