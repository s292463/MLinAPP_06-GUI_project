	2??n86@2??n86@!2??n86@	?^R?T@?^R?T@!?^R?T@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails02??n86@ɐc???1W]?jJ:"@I{ܷZ'?&@YU??-????r0*	T㥛??a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????·??!???.?-@@)???d???1???d6:@:Preprocessing2U
Iterator::Model::ParallelMapV2AEկt>??!w?ou?k3@)AEկt>??1w?ou?k3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap3?`????!>??Z?W8@)???"R??1?????*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipW?Y????!\|???-Q@)?%Tpx??1R?E?(@:Preprocessing2F
Iterator::Model??ʅʿ??!??$?H?@)7?[ A??13:?^w?'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicei㈵???!p/??&@)i㈵???1p/??&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor7??????!?9?n_?@)7??????1?9?n_?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?51.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?^R?T@I???a?J@Q???b??D@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ɐc???ɐc???!ɐc???      ??!       "	W]?jJ:"@W]?jJ:"@!W]?jJ:"@*      ??!       2      ??!       :	{ܷZ'?&@{ܷZ'?&@!{ܷZ'?&@B      ??!       J	U??-????U??-????!U??-????R      ??!       Z	U??-????U??-????!U??-????b      ??!       JGPUY?^R?T@b q???a?J@y???b??D@