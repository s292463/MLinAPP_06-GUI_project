	rn???0@rn???0@!rn???0@	?)??????)?????!?)?????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCrn???0@??ao??1???
-@II?s
????Y? n/??rEagerKernelExecute 0*	?z?GEv@2U
Iterator::Model::ParallelMapV2?U??6o??!?Ģ???J@)?U??6o??1?Ģ???J@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS[? ???!?I?A??4@)???x?@??1#?????2@:Preprocessing2F
Iterator::Model\ A?c???!??Tm?mQ@)?)?t??1S??:%0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??`?d??!mH?Y?	@)??`?d??1mH?Y?	@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??Y?rL??!M????q@)_?R#?3??1-?^>@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipV+~????!??JFH>@)qW?"???1?;?	|?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?*n|?!??xK?*??)?J?*n|?1??xK?*??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS?K?^??!"???@)?:?*?h?1??|????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?)?????I??%??*@Q???r_U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ao????ao??!??ao??      ??!       "	???
-@???
-@!???
-@*      ??!       2      ??!       :	I?s
????I?s
????!I?s
????B      ??!       J	? n/??? n/??!? n/??R      ??!       Z	? n/??? n/??!? n/??b      ??!       JGPUY?)?????b q??%??*@y???r_U@