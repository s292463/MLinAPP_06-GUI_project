	g???ْ@g???ْ@!g???ْ@	?t??>^
@?t??>^
@!?t??>^
@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLg???ْ@it?3??1(5
? @A |(ђ??I}????9	@Y]ݱ?&??rEagerKernelExecute 0*	?n??Rb@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;ŪA??!M?ot?(@@)??C p??1?J?@*;;@:Preprocessing2F
Iterator::Model??AA)Z??!??A??@@)y?@e????1lqU:O3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice>+N???!?N Hbn-@)>+N???1?N Hbn-@:Preprocessing2U
Iterator::Model::ParallelMapV2?CV???!_8?[??,@)?CV???1_8?[??,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatev???_w??!??E?˚8@)[?a/???1????4?#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipa7l[?ٸ?!??)??P@)?'*?T??1a???)?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?L?T?~?!?ǩ??Z@)?L?T?~?1?ǩ??Z@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???jHܣ?!rr
Xv:@)??M~?Nf?1??H?Ǹ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?45.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?t??>^
@I?@e??P@Qw?g???>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	it?3??it?3??!it?3??      ??!       "	(5
? @(5
? @!(5
? @*      ??!       2	 |(ђ?? |(ђ??! |(ђ??:	}????9	@}????9	@!}????9	@B      ??!       J	]ݱ?&??]ݱ?&??!]ݱ?&??R      ??!       Z	]ݱ?&??]ݱ?&??!]ݱ?&??b      ??!       JGPUY?t??>^
@b q?@e??P@yw?g???>@