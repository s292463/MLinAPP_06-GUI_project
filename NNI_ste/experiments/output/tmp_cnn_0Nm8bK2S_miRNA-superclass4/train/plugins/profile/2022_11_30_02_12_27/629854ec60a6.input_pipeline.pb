	??}???3@??}???3@!??}???3@      ??!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??}???3@S??????1?F????1@Ik???u???rEagerKernelExecute 0*	?????yc@2F
Iterator::Modely=???!\<???G@){0)>>!??13??qA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???3????!?A0???>@)G6??1v??ܖ:@:Preprocessing2U
Iterator::Model::ParallelMapV2'????9??!9s:??+@)'????9??19s:??+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?I?U??!???? ?@)?I?U??1???? ?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip'l??ô?!????J@)N?f????1?,????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?Q?(?1??!??ףF?*@)vöE???1"???l?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??Q?z?!?H????@)??Q?z?1?H????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?J???!;???r<.@)???ig?1?Bj?`Y??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIx_~??'@QQ4P?V@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	S??????S??????!S??????      ??!       "	?F????1@?F????1@!?F????1@*      ??!       2      ??!       :	k???u???k???u???!k???u???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qx_~??'@yQ4P?V@