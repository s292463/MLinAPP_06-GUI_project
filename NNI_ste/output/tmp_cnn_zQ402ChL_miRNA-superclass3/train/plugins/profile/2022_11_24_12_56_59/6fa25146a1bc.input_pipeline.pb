	??@?? @??@?? @!??@?? @      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??@?? @?xͫ:k@1???-=??A~8H????Ii5$??4@rEagerKernelExecute 0*	????x%a@2F
Iterator::ModelN?@?C???!,???H@)?????`??1vu?5B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???1v£?!?@?s?"<@)w?ِf??1??LuZ7@:Preprocessing2U
Iterator::Model::ParallelMapV2?k	??g??!??K7?+@)?k	??g??1??K7?+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceR?.?????!ZJ ?i:@)R?.?????1ZJ ?i:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?e?????!?$X?s?+@)???ْU??1???t}?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???_???!???>|I@)e?fb???1w?q???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?i???z?!)r??!@)?i???z?1)r??!@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapi?wak??!???@?/@)=?e?YJf?1??{???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 26.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?52.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIf&???S@Qgf?;A5@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?xͫ:k@?xͫ:k@!?xͫ:k@      ??!       "	???-=?????-=??!???-=??*      ??!       2	~8H????~8H????!~8H????:	i5$??4@i5$??4@!i5$??4@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qf&???S@ygf?;A5@