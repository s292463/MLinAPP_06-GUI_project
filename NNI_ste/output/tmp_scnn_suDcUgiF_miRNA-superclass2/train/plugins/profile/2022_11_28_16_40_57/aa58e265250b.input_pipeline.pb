	L?;??Y?@L?;??Y?@!L?;??Y?@	A??9@A??9@!A??9@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0L?;??Y?@??n,(??1????|W0@Ik??=-)@Y?4?@r0*	~j?t?xc@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatN?@?C???!??ae@@)???[???1Z^?Ĥ4:@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip<0?????!???Tc?Q@)fٓ????1??:_??1@:Preprocessing2U
Iterator::Model::ParallelMapV2v??Sǚ?!z??0@)v??Sǚ?1z??0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???)???! TE)s'6@)9?M?a???1??)@:Preprocessing2F
Iterator::Model?Z&??|??!+9??r2<@)E?D??2??1y4d8?&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?/??乎?!??~9eC#@)?/??乎?1??~9eC#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorgE?D????!6???E@)gE?D????16???E@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?40.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9A??9@IP????(D@QhmkLJ@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??n,(????n,(??!??n,(??      ??!       "	????|W0@????|W0@!????|W0@*      ??!       2      ??!       :	k??=-)@k??=-)@!k??=-)@B      ??!       J	?4?@?4?@!?4?@R      ??!       Z	?4?@?4?@!?4?@b      ??!       JGPUYA??9@b qP????(D@yhmkLJ@