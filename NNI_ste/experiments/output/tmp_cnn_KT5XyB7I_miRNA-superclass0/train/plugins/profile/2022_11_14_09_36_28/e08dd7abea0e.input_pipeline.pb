	?k?ճ!@?k?ճ!@!?k?ճ!@	%lA?U??%lA?U??!%lA?U??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?k?ճ!@???tx(@1?u????A????B??I8h?>?@Y??y7??rEagerKernelExecute 0*	,???e@2Z
#Iterator::Model::ParallelMapV2::Zip?	?c??!?T;N?Q@)???'???1???}??;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ȳ˧?!????;@)??5>????1ud??'7@:Preprocessing2F
Iterator::Modely???????!???ǚ=@)QS?'???1?́&1@:Preprocessing2U
Iterator::Model::ParallelMapV2??xxρ??!:???j?(@)??xxρ??1:???j?(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice겘?|\??!??#X?@)겘?|\??1??#X?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?2#???!6:7?*@)??m????1?ucKz@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?Oq~?!k?u?Y?@)?Oq~?1k?u?Y?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???Hh˙?!Z??2??-@)?4-?2j?1%??gX??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 27.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?53.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9&lA?U??I???/>T@Q?f?B?1@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???tx(@???tx(@!???tx(@      ??!       "	?u?????u????!?u????*      ??!       2	????B??????B??!????B??:	8h?>?@8h?>?@!8h?>?@B      ??!       J	??y7????y7??!??y7??R      ??!       Z	??y7????y7??!??y7??b      ??!       JGPUY&lA?U??b q???/>T@y?f?B?1@