	?ɧw@?ɧw@!?ɧw@	b6???@b6???@!b6???@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?ɧw@?'?ڑ??1?E?x?@ALU?????IaS?Q?_@Y??	?Y???rEagerKernelExecute 0*~j?t??f@)       =2F
Iterator::Model!%̴???!??????I@)?G?????1?n???*C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat)&o?????![G?I?:@)?9"ߥԥ?10	+???7@:Preprocessing2U
Iterator::Model::ParallelMapV23??????!UC??È)@)3??????1UC??È)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice^??????!??Kcb?#@)^??????1??Kcb?#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip5?+-#???!K@w+sH@)Wj1x???1???Sm@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatey#????!]????-@)6?:???1*??`?"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorbg
??x?![?9??t
@)bg
??x?1[?9??t
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?y????!B?d??S0@)?B???d?1%\E??C??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?37.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9b6???@I?)W??L@Q?/W?C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?'?ڑ???'?ڑ??!?'?ڑ??      ??!       "	?E?x?@?E?x?@!?E?x?@*      ??!       2	LU?????LU?????!LU?????:	aS?Q?_@aS?Q?_@!aS?Q?_@B      ??!       J	??	?Y?????	?Y???!??	?Y???R      ??!       Z	??	?Y?????	?Y???!??	?Y???b      ??!       JGPUYb6???@b q?)W??L@y?/W?C@