	zo?!"@zo?!"@!zo?!"@	?.?@@@?.?@@@!?.?@@@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCzo?!"@?%?2???1?????7@IQ??????Y?Xİ???rEagerKernelExecute 0*	`??"?ed@2F
Iterator::ModelkIG9?M??!k??4?LH@)???????1InM} ?A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatK#f?y???!?N[????@)?????1j|????;@:Preprocessing2U
Iterator::Model::ParallelMapV2h?????!????S*@)h?????1????S*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice\?=????! v??W@)\?=????1 v??W@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??6?x??!?A?	?I@)??~j?t??1\?O?gI@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateh׿???!??,???'@)t]???ԁ?1TI???W@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor]n0?a?{?!wH?FHx@)]n0?a?{?1wH?FHx@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??乾??!?ƥ?%?+@)?rL?i?1N7??^??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?17.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?.?@@@I?h]Ek?5@QU??)??R@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?%?2????%?2???!?%?2???      ??!       "	?????7@?????7@!?????7@*      ??!       2      ??!       :	Q??????Q??????!Q??????B      ??!       J	?Xİ????Xİ???!?Xİ???R      ??!       Z	?Xİ????Xİ???!?Xİ???b      ??!       JGPUY?.?@@@b q?h]Ek?5@yU??)??R@