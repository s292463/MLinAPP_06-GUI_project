	?????,@?????,@!?????,@	? ??@? ??@!? ??@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?????,@1'h??'m?1^?? @I(?.??T'@Y{?"0ַ??r0*	?ʡE?/d@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat,}??????!?J:?FE@)????H??1??ԐOPB@:Preprocessing2F
Iterator::ModelhY??????!?L?Z?>@)"r?z?f??1l?????.@:Preprocessing2U
Iterator::Model::ParallelMapV2?@??Lj??!K??Q?-@)?@??Lj??1K??Q?-@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6?Ko.??!?LM?5@)__?R#???10?!(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceq???h??!?????#@)q???h??1?????#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?N?z1???!?!,??@)?N?z1???1?!,??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???6???!?,L?xQ@)?tx㧁?1i???Z@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?81.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9? ??@I?8????T@Q4V~?-@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	1'h??'m?1'h??'m?!1'h??'m?      ??!       "	^?? @^?? @!^?? @*      ??!       2      ??!       :	(?.??T'@(?.??T'@!(?.??T'@B      ??!       J	{?"0ַ??{?"0ַ??!{?"0ַ??R      ??!       Z	{?"0ַ??{?"0ַ??!{?"0ַ??b      ??!       JGPUY? ??@b q?8????T@y4V~?-@