	???G? @???G? @!???G? @	v?ޑ??@v?ޑ??@!v?ޑ??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???G? @?'?XQ??1?F? ?@I?e3??v@Y????n???rEagerKernelExecute 0*	ˡE???g@2F
Iterator::Modelj1x??͵?!`??\?oF@)X?2ı.??1rc?H??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???I???!&z?kI8@)_B?D??1?>?v?4@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????}ɺ?!?p5???K@)t#,*?t??1?W v??2@:Preprocessing2U
Iterator::Model::ParallelMapV2?0{?vښ?!?v?ᶡ+@)?0{?vښ?1?v?ᶡ+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicev??$?p??!?u??@)v??$?p??1?u??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?%?"?d??!഑?.?$@)??? ?X??1??5??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?u?ݑ?z?!???v?w@)?u?ݑ?z?1???v?w@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???Y???!?k?'@)P?<?e?1fO?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?27.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9w?ޑ??@Ih?zB??>@QZR?*exP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?'?XQ???'?XQ??!?'?XQ??      ??!       "	?F? ?@?F? ?@!?F? ?@*      ??!       2      ??!       :	?e3??v@?e3??v@!?e3??v@B      ??!       J	????n???????n???!????n???R      ??!       Z	????n???????n???!????n???b      ??!       JGPUYw?ޑ??@b qh?zB??>@yZR?*exP@