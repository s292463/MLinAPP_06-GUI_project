	??|?R4@??|?R4@!??|?R4@	?Q1??p @?Q1??p @!?Q1??p @"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??|?R4@?ND???1T??7??@Iv??ݰu)@Y&VF#?W??r0*	?|?5^?g@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????w??!ntU?R@@)uF^?Ī?19?ظ?\;@:Preprocessing2U
Iterator::Model::ParallelMapV2?$?@??!?˒?1@)?$?@??1?˒?1@:Preprocessing2F
Iterator::Model??^???!8q4?DA@)?c?~???1??Rog?0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap* ??q??!m???#?3@)??-@ۚ?1 ???s+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipL5????!d?eq?]P@)???E_A??1???x?*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice???Mb??!?]uh	?@)???Mb??1?]uh	?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??z?p̂?!?*I17@)??z?p̂?1?*I17@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?63.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?Q1??p @I??䙓?O@Q?Y?y*<@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ND????ND???!?ND???      ??!       "	T??7??@T??7??@!T??7??@*      ??!       2      ??!       :	v??ݰu)@v??ݰu)@!v??ݰu)@B      ??!       J	&VF#?W??&VF#?W??!&VF#?W??R      ??!       Z	&VF#?W??&VF#?W??!&VF#?W??b      ??!       JGPUY?Q1??p @b q??䙓?O@y?Y?y*<@