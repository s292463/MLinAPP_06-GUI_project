	Y?n}??@Y?n}??@!Y?n}??@	?Xp????Xp???!?Xp???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLY?n}??@N??}??1???=@A?BY??Z??IZ??!ŨM@Y??|???	@rEagerKernelExecute 0*	i??|??b@2F
Iterator::Model	3?z???!??L??I@)?	?O????1E+?N@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?=?-??!??w?6E>@)?Hg`?e??1??? ??;@:Preprocessing2U
Iterator::Model::ParallelMapV2?T?]??!J5z??2@)?T?]??1J5z??2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?B]?!???Q?1'@)KZ??φ?1+?|I??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??F?????!:??p2H@)??]???1???.?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??%jjy?!Q5?Y?@)??%jjy?1Q5?Y?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapya?X5??!樴?V??@)?3?ۃ`?1??ͳ????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensork*??.?^?!D?co????)k*??.?^?1D?co????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice??E??\Z?!????/7??)??E??\Z?1????/7??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?10.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?Xp???I?01ឺ%@Q?)???$V@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	N??}??N??}??!N??}??      ??!       "	???=@???=@!???=@*      ??!       2	?BY??Z???BY??Z??!?BY??Z??:	Z??!ŨM@Z??!ŨM@!Z??!ŨM@B      ??!       J	??|???	@??|???	@!??|???	@R      ??!       Z	??|???	@??|???	@!??|???	@b      ??!       JGPUY?Xp???b q?01ឺ%@y?)???$V@