	?1?Mc+$@?1?Mc+$@!?1?Mc+$@	??;?????;???!??;???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?1?Mc+$@v??~@1?s(CU??A?x?Z????I`????s@Y????rEagerKernelExecute 0*	U㥛?2s@2U
Iterator::Model::ParallelMapV2?8b->??!}xL? L@)?8b->??1}xL? L@:Preprocessing2F
Iterator::Model?֍wG??!???<"?Q@)yxρ???1/????/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?gx????!?pLXgY)@)???̯???1hXw&~%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate@/ܹ0қ?!73???!@)[?[!?ƒ?1??6???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipe??????!??!w<@)L???<Ց?1?1????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?)1	??!?9^,/@)?)1	??1?9^,/@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensors?4?Bx?!???	???)s?4?Bx?1???	???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg?ba????!^?r?v#@)46<?Rf?1j?X.Nc??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 25.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?55.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??;???I?P?kUT@QAp|?a1@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	v??~@v??~@!v??~@      ??!       "	?s(CU???s(CU??!?s(CU??*      ??!       2	?x?Z?????x?Z????!?x?Z????:	`????s@`????s@!`????s@B      ??!       J	????????!????R      ??!       Z	????????!????b      ??!       JGPUY??;???b q?P?kUT@yAp|?a1@