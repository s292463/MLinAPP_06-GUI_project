	?1w-!?#@?1w-!?#@!?1w-!?#@      ??!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?1w-!?#@??֪]??1__?R#?@I???d?f@rEagerKernelExecute 0*	?~j?t[c@2F
Iterator::ModelV?F摳?!??????H@)? ?K???1"???K?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?G?z??!?????;@)??&????1??B?¤7@:Preprocessing2U
Iterator::Model::ParallelMapV2v?S???!׭?>	D2@)v?S???1׭?>	D2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??J?8??!r?I??!@)??J?8??1r?I??!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate$bJ$?˘?!?`?F/@)?????^??1??,X'?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?̔????!cQxUQI@) o?ŏ??1??D:&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorM1AG?z?!???g?@)M1AG?z?1???g?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??Z??!??????1@)??ŉ?vd?1?H28???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?29.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???	g?=@Q??=f?Q@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??֪]????֪]??!??֪]??      ??!       "	__?R#?@__?R#?@!__?R#?@*      ??!       2      ??!       :	???d?f@???d?f@!???d?f@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???	g?=@y??=f?Q@