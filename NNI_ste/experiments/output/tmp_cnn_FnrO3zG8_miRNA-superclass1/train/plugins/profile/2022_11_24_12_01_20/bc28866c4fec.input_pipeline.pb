	?7U?@?7U?@!?7U?@	Y8D??@Y8D??@!Y8D??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?7U?@Y??w???1??ܵ@AV??L???I????_?@Yk-?B;???rEagerKernelExecute 0*	/??ّ@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??>x??!???J@)?*??p??1b???8I@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map˅ʿ?W??!?$??aC@)?Ǵ6????1???D??A@:Preprocessing2F
Iterator::ModelYL?Q??!?,?kR@)?[!????1u?U0@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat???HLP??!???Nj
@);S???.??1?s???@:Preprocessing2U
Iterator::Model::ParallelMapV2???c"??!??}̑???)???c"??1??}̑???:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??'*֔?!od?8=??)Y???tw??1"??.p&??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateEׅ?O??!9?Q?0??)^??????1ɮv3???:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::PrefetchМ?)?d??!G?<?????)М?)?d??1G?<?????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip@mT?Y??!y??'?K@)U?G???|?1?l????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceȷw??{?!*?????)ȷw??{?1*?????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(?XQ?ix?!?e????)(?XQ?ix?1?e????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?,^,q?!?????P??)?,^,q?1?????P??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?-v??2??!r????A??)?F?ҿ$e?1 ?W?????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensort^c???J?!]?-??_??)t^c???J?1]?-??_??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?40.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Y8D??@IZ?"W?L@Q 0???C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Y??w???Y??w???!Y??w???      ??!       "	??ܵ@??ܵ@!??ܵ@*      ??!       2	V??L???V??L???!V??L???:	????_?@????_?@!????_?@B      ??!       J	k-?B;???k-?B;???!k-?B;???R      ??!       Z	k-?B;???k-?B;???!k-?B;???b      ??!       JGPUYY8D??@b qZ?"W?L@y 0???C@