	3??A??@3??A??@!3??A??@	?/$?1l@?/$?1l@!?/$?1l@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL3??A??@YR?>????1c*????@A????u6??IC??g@Yb??????rEagerKernelExecute 0*	?Zd;?r@2F
Iterator::Modelp'?_??!~???nQ@)?x?0DN??1ӊQI;N@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat	?c???!g?w`1@)?????X??1??
? .@:Preprocessing2U
Iterator::Model::ParallelMapV2?u??ݰ??!????N(#@)?u??ݰ??1????N(#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?7/N|???!?#? o@)?7/N|???1?#? o@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?L!u??!???aE>@) ?? ??1?L???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?cϞ??!юBM?!@)t(CUL??1?C?3k@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor7 !?|?!??L8]?@)7 !?|?1??L8]?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapen?ݳ??!?F	l?#@)cb?qm?h?1?|k\????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 14.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?30.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?/$?1l@I5J????F@Q?r????I@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	YR?>????YR?>????!YR?>????      ??!       "	c*????@c*????@!c*????@*      ??!       2	????u6??????u6??!????u6??:	C??g@C??g@!C??g@B      ??!       J	b??????b??????!b??????R      ??!       Z	b??????b??????!b??????b      ??!       JGPUY?/$?1l@b q5J????F@y?r????I@