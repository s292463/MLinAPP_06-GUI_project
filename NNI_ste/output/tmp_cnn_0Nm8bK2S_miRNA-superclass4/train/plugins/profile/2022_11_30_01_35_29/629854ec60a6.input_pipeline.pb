	?5Z?(&@?5Z?(&@!?5Z?(&@	mg????mg????!mg????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?5Z?(&@???h????1??"??!@IM??????Y?\??'??rEagerKernelExecute 0*	?x?&1?w@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateu?? ???!>?"??!L@)?tB????1?DY?J@:Preprocessing2F
Iterator::Model??я?S??!?(?xR;@)b?[>????1'^wmF3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??I~į??!??]՞)@)z?I|???1??R?/?%@:Preprocessing2U
Iterator::Model::ParallelMapV2eQ?E???!???1 @)eQ?E???1???1 @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?i2?m???!???<k?@)?i2?m???1???<k?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????ׁ??!?u?a+R@)??-$`??1?Ũ<u%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor8?a?A
~?!:Y *-??)8?a?A
~?1:Y *-??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapRD?U????!??z`??L@)m7?7M?m?1d?	?)???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?15.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9mg????I\?S?/'2@Q5??t- T@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???h???????h????!???h????      ??!       "	??"??!@??"??!@!??"??!@*      ??!       2      ??!       :	M??????M??????!M??????B      ??!       J	?\??'???\??'??!?\??'??R      ??!       Z	?\??'???\??'??!?\??'??b      ??!       JGPUYmg????b q\?S?/'2@y5??t- T@