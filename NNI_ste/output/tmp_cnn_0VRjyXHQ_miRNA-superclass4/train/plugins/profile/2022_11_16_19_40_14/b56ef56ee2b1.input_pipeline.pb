	ŏ1w-?@ŏ1w-?@!ŏ1w-?@	hDw4VQ@hDw4VQ@!hDw4VQ@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLŏ1w-?@?m??)??1˃?9@A?h?x?J??I??tB@Y?쟧???rEagerKernelExecute 0*	??Q??d@2F
Iterator::Model}v?uŌ??!?u?/?H@)??_=?[??1'}?X7A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?1%????!W,Ջ??9@)?j???u??1^?{?5@:Preprocessing2U
Iterator::Model::ParallelMapV26 B\9{??!?s??+@)6 B\9{??1?s??+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?:?zj??!?B?)@)?:?zj??1?B?)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???????!??G???3@))%?????1?o@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?6?4D??!W?#?(?I@)幾	??1?觞?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;ŪA?{?!9?@|.@);ŪA?{?19?@|.@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapͬ??????!?wj?5@)c`?e?1?83|???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?35.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9iDw4VQ@I7?Vل
K@Q;U`P?D@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?m??)???m??)??!?m??)??      ??!       "	˃?9@˃?9@!˃?9@*      ??!       2	?h?x?J???h?x?J??!?h?x?J??:	??tB@??tB@!??tB@B      ??!       J	?쟧????쟧???!?쟧???R      ??!       Z	?쟧????쟧???!?쟧???b      ??!       JGPUYiDw4VQ@b q7?Vل
K@y;U`P?D@