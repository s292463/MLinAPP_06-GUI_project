	?"??)8!@?"??)8!@!?"??)8!@	l?CЎ@l?CЎ@!l?CЎ@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?"??)8!@?ʦ\???1??9?@??ALp??;??Ir6??@Y?T??????rEagerKernelExecute 0*	F????d@2F
Iterator::Model3nj?????!/?Z	?G@)W??????1??_'?G?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatj??4ӽ??!?K?n?;@)ղ??Hh??1F??c??7@:Preprocessing2U
Iterator::Model::ParallelMapV2?S???!0{N?Qr0@)?S???10{N?Qr0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????A{??!??(??"J@)l??TO???1?m???g#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??ݯ|??!D?Ւ@)??ݯ|??1D?Ւ@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate:??)??!3,??m?*@)??'*ք?1O?'Z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?<??S?z?!?.?:@)?<??S?z?1?.?:@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??8~??!S>????-@)?[?O?b?1?X8????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?54.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9l?CЎ@I[v;S?S@Q??	??;5@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ʦ\????ʦ\???!?ʦ\???      ??!       "	??9?@????9?@??!??9?@??*      ??!       2	Lp??;??Lp??;??!Lp??;??:	r6??@r6??@!r6??@B      ??!       J	?T???????T??????!?T??????R      ??!       Z	?T???????T??????!?T??????b      ??!       JGPUYl?CЎ@b q[v;S?S@y??	??;5@