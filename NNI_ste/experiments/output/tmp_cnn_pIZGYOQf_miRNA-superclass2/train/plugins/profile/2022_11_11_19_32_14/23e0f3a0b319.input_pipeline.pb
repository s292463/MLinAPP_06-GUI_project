	???G?@???G?@!???G?@	????@????@!????@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???G?@k*?®??1????<@A??fH??I??9???@Y?|x? #??rEagerKernelExecute 0*	fffffV?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?>:u?3??!???<H@)c??Ր8??1ɀz?r?F@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?=Ab????!4??t?F@)???ϝ`??1???4?D@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?ɧǶ??!?x$??@)˟o???1????@:Preprocessing2F
Iterator::Model?u?T??!;9??@)??t_Μ?1@?9?-@:Preprocessing2U
Iterator::Model::ParallelMapV2(5
If??!kq??}??)(5
If??1kq??}??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatef???8??!?OBL????)?Ws?`???1??ܽ@W??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat:=?Ƃ??!??}???)l????߉?1?u?i9??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?=$|?o??!???U???)?=$|?o??1???U???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip,??? ??!?յ??LI@)??ǵ?b|?1??L???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???/fKv?!??"(????)???/fKv?1??"(????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?<+i?7t?!&k.?????)?<+i?7t?1&k.?????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeȗP??q?!??????)ȗP??q?1??????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate0? ???{?!0?Qġ???)?yUg??^?1x???x??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?A?p?-N?!z?X???)?A?p?-N?1z?X???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?36.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9????@I???K@Q?ǜ f?C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	k*?®??k*?®??!k*?®??      ??!       "	????<@????<@!????<@*      ??!       2	??fH????fH??!??fH??:	??9???@??9???@!??9???@B      ??!       J	?|x? #???|x? #??!?|x? #??R      ??!       Z	?|x? #???|x? #??!?|x? #??b      ??!       JGPUY????@b q???K@y?ǜ f?C@