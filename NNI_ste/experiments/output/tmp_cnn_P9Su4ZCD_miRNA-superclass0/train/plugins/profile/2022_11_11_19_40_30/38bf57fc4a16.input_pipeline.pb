	?g@??$@?g@??$@!?g@??$@	???A@???A@!???A@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?g@??$@?66;R}??1?DR??@A???????I??	??#@Y??s?v???rEagerKernelExecute 0*	??Mb???@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapw?T????!?K?&L@)T??b???1??J@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?-]?6??!T6?*?1<@)??pvk???1o-??J?6@:Preprocessing2F
Iterator::Model?<?????!/g???+@)#?ng_y??1l???'@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatn???Wu??!?#?(2~@)???C?X??1???3@:Preprocessing2U
Iterator::Model::ParallelMapV2??5|??!?֒?}??)??5|??1?֒?}??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??-Y??!??DoU???)??e?O7??1?RFq????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?e?%⭓?!?2?????)噗????1M3.C??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?y??0???!yM?(????)?y??0???1yM?(????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?.??????!
??b?L@)?4-?2z?1F?.jH???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?˵h?v?!M?;r????)?˵h?v?1M?;r????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice??y7r?!{???5??)??y7r?1{???5??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range???q??q?!c0??±??)???q??q?1c0??±??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate 
fL?z?!whT{U'??)?6?ُa?1?Uwk???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor{?\?&?K?!???ߟ(??){?\?&?K?1???ߟ(??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?24.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???A@Id???K1D@Q???kK@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?66;R}???66;R}??!?66;R}??      ??!       "	?DR??@?DR??@!?DR??@*      ??!       2	??????????????!???????:	??	??#@??	??#@!??	??#@B      ??!       J	??s?v?????s?v???!??s?v???R      ??!       Z	??s?v?????s?v???!??s?v???b      ??!       JGPUY???A@b qd???K1D@y???kK@