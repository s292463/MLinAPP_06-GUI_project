	ٱ?ץ,@ٱ?ץ,@!ٱ?ץ,@	)/?\??@)/?\??@!)/?\??@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ٱ?ץ,@?&?5?P?1???c;@I7 !F$@Y??????r0*	R???ic@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?A
?B???!?<?ƯB@)?q?Z|
??1*??F?;>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?C?????!?C?Wxq:@)G????g??1?
?b?3@:Preprocessing2U
Iterator::Model::ParallelMapV2 ??q???!*-???0@) ??q???1*-???0@:Preprocessing2F
Iterator::Model??q6??!)V??'??@)t)?*????1?Q?a?
/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??I'L??!???V?@)??I'L??1???V?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???1?3??!v*??Q@)ȶ8Kɂ?1??? ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????!ȡ??V@)?????1ȡ??V@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?70.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9)/?\??@I???ɴ?Q@Q?̈A??7@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?&?5?P??&?5?P?!?&?5?P?      ??!       "	???c;@???c;@!???c;@*      ??!       2      ??!       :	7 !F$@7 !F$@!7 !F$@B      ??!       J	????????????!??????R      ??!       Z	????????????!??????b      ??!       JGPUY)/?\??@b q???ɴ?Q@y?̈A??7@