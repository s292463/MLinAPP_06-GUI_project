	DkE??4!@DkE??4!@!DkE??4!@      ??!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:DkE??4!@cFx{??1?7?nz@I??w?G/@rEagerKernelExecute 0*	??n?Xv@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateT:X??0??!? ?^4?M@)E?a????1???U?J@:Preprocessing2F
Iterator::Model????	???!v??韪6@)`YiR
???1???
6=0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??uoEb??!?k?q?,@)??FtϺ??1w??e?(@:Preprocessing2U
Iterator::Model::ParallelMapV2?+ٱ???!??{??@)?+ٱ???1??{??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicesh??|???!?y??6@)sh??|???1?y??6@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??f????!cY?XUS@)IIC????1?5\??c@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;?vٯ;}?!s(?_???);?vٯ;}?1s(?_???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap]??J???!w-???N@):?Y?Xh?1.??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?47.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?(8???H@Q4??eI@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	cFx{??cFx{??!cFx{??      ??!       "	?7?nz@?7?nz@!?7?nz@*      ??!       2      ??!       :	??w?G/@??w?G/@!??w?G/@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?(8???H@y4??eI@