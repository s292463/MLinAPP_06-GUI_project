	?ފ?"@?ފ?"@!?ފ?"@	???9|=?????9|=??!???9|=??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?ފ?"@v??@1$?jf-???Ab???4??I?v???@Y???@???rEagerKernelExecute 0*?C?lg??@)      P=2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??i??+??!?A? ?G@)%̴?+???1.?w??E@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map[?Qf???!??Ő?F@)???rf??1{?6
??E@:Preprocessing2F
Iterator::Model5^?I??!??X?_0@)?Nϻ????1z?mg(
@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?6+1Ϣ?!H ?[?@)3?,%?I??1Շ-??L@:Preprocessing2U
Iterator::Model::ParallelMapV2??J??ƚ?!?؆??p??)??J??ƚ?1?؆??p??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice??q?@H??!? ????)??q?@H??1? ????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???5??!w?ҭJr??)?? v???1???eI???:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ʡE????!??M????)??ޫV&??1b??????:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?À%W??!???j??)?À%W??1???j??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?~?Ϛ??!?^LCc?H@) ????}?1???u???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor`"?:?vy?!??!??)`"?:?vy?1??!??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeS?r/0+t?!V???k??)S?r/0+t?1V???k??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate??p?Qe??!???V???)hY????`?1p?-?s???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorK?8???L?!?(`W????)K?8???L?1?(`W????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 27.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?57.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???9|=??I?**I(WU@Q?rw/?)@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	v??@v??@!v??@      ??!       "	$?jf-???$?jf-???!$?jf-???*      ??!       2	b???4??b???4??!b???4??:	?v???@?v???@!?v???@B      ??!       J	???@??????@???!???@???R      ??!       Z	???@??????@???!???@???b      ??!       JGPUY???9|=??b q?**I(WU@y?rw/?)@