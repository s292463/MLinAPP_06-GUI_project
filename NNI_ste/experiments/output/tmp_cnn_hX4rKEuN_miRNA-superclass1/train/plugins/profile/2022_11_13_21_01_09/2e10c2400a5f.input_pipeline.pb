	R??B@R??B@!R??B@	?Ĺs%?@?Ĺs%?@!?Ĺs%?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLR??B@?P????1????av??A????K??I*V???@Y??HK????rEagerKernelExecute 0*	^?IGe@2F
Iterator::Model?a?1????! *nؔD@)?Iط???1??i?<3<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?7??d???!?
A0AM<@)????G???1?,L??7@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?{?Y?H??!ɩIW?*@)?{?Y?H??1ɩIW?*@:Preprocessing2U
Iterator::Model::ParallelMapV2???5???!z1???)@)???5???1z1???)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???????! ???'kM@)5?؀??1ҙ?R?#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??Z}uU??!??o???2@)??gB?Ă?1i?+I܈@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorS???t??!?x???`@)S???t??1?x???`@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapn?HJz??!?o|d?4@)????L0l?1? @, @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?51.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?Ĺs%?@Ir?N8OlO@Q?:̋@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?P?????P????!?P????      ??!       "	????av??????av??!????av??*      ??!       2	????K??????K??!????K??:	*V???@*V???@!*V???@B      ??!       J	??HK??????HK????!??HK????R      ??!       Z	??HK??????HK????!??HK????b      ??!       JGPUY?Ĺs%?@b qr?N8OlO@y?:̋@@