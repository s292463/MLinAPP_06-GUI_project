	????=@????=@!????=@	X?5]YR@X?5]YR@!X?5]YR@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????=@???*P???1????0@I?Ȳ`?w&@Y??@J???r0*	??ʡ??@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??:8?#@!?>??;V@)??D?"@1?{95?V@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?k	??'??!??q??$@)??3K???1C?9???#@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??XP???!8Dì????)?&??????1?F?{Q???:Preprocessing2F
Iterator::Model??L?????!?K??????)i??????18??????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???Wy??!n?0?ǐ??)?;??J"??1???????:Preprocessing2U
Iterator::Model::ParallelMapV2????w???!H??rW???)????w???1H??rW???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipI?????!?3?,?%@)2<??X???1?}BP???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorkׄ?Ơ??!fB??????)kׄ?Ơ??1fB??????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice2 Tq???!{&at?H??)2 Tq???1{&at?H??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetchy\T??b??!o^(v??)y\T??b??1o^(v??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangeܸ????t?!??߈?L??)ܸ????t?1??߈?L??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice????Wa?!4T???>??)????Wa?14T???>??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?37.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Y?5]YR@I|X??C@Q??????K@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???*P??????*P???!???*P???      ??!       "	????0@????0@!????0@*      ??!       2      ??!       :	?Ȳ`?w&@?Ȳ`?w&@!?Ȳ`?w&@B      ??!       J	??@J?????@J???!??@J???R      ??!       Z	??@J?????@J???!??@J???b      ??!       JGPUYY?5]YR@b q|X??C@y??????K@