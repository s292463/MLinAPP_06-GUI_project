	???o>(@???o>(@!???o>(@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???o>(@ؼ??Z @18-x?W?@Ah??|?5??I??3.@rEagerKernelExecute 0*	???Q\e@2F
Iterator::Model???*?w??!Ƭ??)@F@)???9????1??Ip??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???j?=??!"`?`k9@)?'L͢?1?!?}5@:Preprocessing2U
Iterator::Model::ParallelMapV2?J?8????!b????)@)?J?8????1b????)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??bc^G??!:SKֿK@)?Y,E???1?p??b)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicen??)"??!5V?V?%@)n??)"??15V?V?%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate*?TPQ???!?+?No?/@)xcAaP???1,?ц1,@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?W<?H?{?!t@??r@)?W<?H?{?1t@??r@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?F??1???!???@??1@)|????e?1????9???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 17.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?59.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI@?-z?pS@Q oI?<6@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ؼ??Z @ؼ??Z @!ؼ??Z @      ??!       "	8-x?W?@8-x?W?@!8-x?W?@*      ??!       2	h??|?5??h??|?5??!h??|?5??:	??3.@??3.@!??3.@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@?-z?pS@y oI?<6@