	]ݱ?&? @]ݱ?&? @!]ݱ?&? @	y??#1?@y??#1?@!y??#1?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL]ݱ?&? @?I???1?i? ??@A?q4GV~??IW%?}?%@Y?~K?|??rEagerKernelExecute 0*	rh??|U?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?}?????!(e??S@)?(?????1?J??ItS@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapy?ՏM???!???0d?+@)ҧU??f??1,??? ?&@:Preprocessing2F
Iterator::ModelD?K?KƱ?!b??n?z@)?=?#d??1?P\??@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?? n/??!?i???@)y?ՏM???1??J1~@:Preprocessing2U
Iterator::Model::ParallelMapV2N} y?P??!??r?g???)N} y?P??1??r?g???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate{-??1??!1?????)m???5???1???
??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip(?4???!????T@)??r۾G??1????D%??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?e3???!|?.??)?????1??س????:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch|E?^??!??Ak3???)|E?^??1??Ak3???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice{?"0?7??!??TK??){?"0?7??1??TK??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?`⏢?|?!B??????)?`⏢?|?1B??????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeV?F?q?!??4???)V?F?q?1??4???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate'i????![?-=???)1?䠄i?1?#v^???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?ص?ݒL?! qax?}??)?ص?ݒL?1 qax?}??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?48.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9y??#1?@I??v???P@Q6?????>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?I????I???!?I???      ??!       "	?i? ??@?i? ??@!?i? ??@*      ??!       2	?q4GV~???q4GV~??!?q4GV~??:	W%?}?%@W%?}?%@!W%?}?%@B      ??!       J	?~K?|???~K?|??!?~K?|??R      ??!       Z	?~K?|???~K?|??!?~K?|??b      ??!       JGPUYy??#1?@b q??v???P@y6?????>@