	$??"?~@$??"?~@!$??"?~@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC$??"?~@[D?7???1???{@A???????I???s?eH@rEagerKernelExecute 0*	?Zd;?a@2F
Iterator::Model+?򑔰?!I?????F@)??r-Z???1
t??{?>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenatey;?i????!?TR%!?>@)?]/M???1l?? R?<@:Preprocessing2U
Iterator::Model::ParallelMapV2???p?Q??!?N@-@)???p?Q??1?N@-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?_???ܳ?!?END>@K@)1??c?g??1??????$@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatTƿϸp??!1yA??&@)??ڦx\??1ŗ9???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor؃I??	y?!?Z?`-@)؃I??	y?1?Z?`-@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????!{??L?e@@)&:?,B?e?1???FW???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?uʣ[?!R?$**???)?uʣ[?1R?$**???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor??W9?Y?!???ƶ??)??W9?Y?1???ƶ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?T??a~$@QiU#?3pV@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	[D?7???[D?7???![D?7???      ??!       "	???{@???{@!???{@*      ??!       2	??????????????!???????:	???s?eH@???s?eH@!???s?eH@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?T??a~$@yiU#?3pV@