	r?@H1@r?@H1@!r?@H1@	?$????#@?$????#@!?$????#@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0r?@H1@~t??gyn?1-??DJ?@I?̒ 5?$@Ytys?V;??r0*	k?t??Y?@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??4?@!??l?\?M@)?0???@1o?6'?L@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap>ʈ@c??!??'?4+C@)?U??f??1J:?w??B@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?E?????!?U0{????)ADj??4??1|?{?;??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat[?*?MF??!6??K"???)??T2 T??1aF?{????:Preprocessing2F
Iterator::Model??"?tu??!???ʑ???)\??Mٙ?1s@	?=N??:Preprocessing2U
Iterator::Model::ParallelMapV2??u????!:(凨???)??u????1:(凨???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipQ?v0b_??!?g?@?C@)?(??{??1?س?????:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?E&??H??!??k??)?E&??H??1??k??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?[?~l??!X???s???)?[?~l??1X???s???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlicei??Iw?!?n?D???)i??Iw?1?n?D???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range5??-</u?!?EK????)5??-</u?1?EK????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?P?yb?!??*?EN??)?P?yb?1??*?EN??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?61.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?$????#@I*M???N@QlS#??<@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	~t??gyn?~t??gyn?!~t??gyn?      ??!       "	-??DJ?@-??DJ?@!-??DJ?@*      ??!       2      ??!       :	?̒ 5?$@?̒ 5?$@!?̒ 5?$@B      ??!       J	tys?V;??tys?V;??!tys?V;??R      ??!       Z	tys?V;??tys?V;??!tys?V;??b      ??!       JGPUY?$????#@b q*M???N@ylS#??<@