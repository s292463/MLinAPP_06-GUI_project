	Swe?@Swe?@!Swe?@	S?3?1?@S?3?1?@!S?3?1?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLSwe?@Z???
??1?V? @A???8???Igd??S@Y=?බ???rEagerKernelExecute 0*	?n??o?@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??S??!??|??G@)??\????1?nbkG@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map`??Ù??!?!?G@)??U????1??/y?A@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat????[??!lY??sJ'@)?֊6ǹ??1oA'¡&@:Preprocessing2F
Iterator::Model?)?TP??!)ԑ??^
@)NB?!???1w?#l&??:Preprocessing2U
Iterator::Model::ParallelMapV2?;jL????!?? Ӗ??)?;jL????1?? Ӗ??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateꗈ?ο??!?(??u???)X?\T??1?R??qG??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatW??Ma???!??o?yl??)Lk??^??1l7?v!??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?IbI??|?!%???21??)?IbI??|?1%???21??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip,?9$????!?????H@)-C??6z?1???N??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??;??~v?!^?P??n??)??;??~v?1^?P??n??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceܸ????t?!???? ???)ܸ????t?1???? ???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??>t?!??bz4??)??>t?1??bz4??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenatek??????!g
?K????)?Y,E?e?1^?2Q????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor????GJ?!?b??E`??)????GJ?1?b??E`??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?47.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9S?3?1?@I?}.{?P@Q???-?=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Z???
??Z???
??!Z???
??      ??!       "	?V? @?V? @!?V? @*      ??!       2	???8??????8???!???8???:	gd??S@gd??S@!gd??S@B      ??!       J	=?බ???=?බ???!=?බ???R      ??!       Z	=?බ???=?බ???!=?බ???b      ??!       JGPUYS?3?1?@b q?}.{?P@y???-?=@