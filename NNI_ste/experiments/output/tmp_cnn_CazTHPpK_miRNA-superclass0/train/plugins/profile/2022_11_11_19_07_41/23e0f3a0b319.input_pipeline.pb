	N??e#@N??e#@!N??e#@	??0??????0????!??0????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLN??e#@?????@1	Q????A??S????I??>Ȳ @Yt$???~??rEagerKernelExecute 0*	??"??B`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??E?T???!??$??S@@)E,b?aL??1Z?|?x;@:Preprocessing2F
Iterator::Model???'??!???2K?E@)5?;???1??\'q*8@:Preprocessing2U
Iterator::Model::ParallelMapV2??$???!!u6>%?3@)??$???1!u6>%?3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?#?@???!???&??"@)?#?@???1???&??"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate5??a0??!2?e#0@)???D??1??q?gl@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipTS?u8???!c6ʹL@)??N}?1\???+?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????{?!??~=f?@)?????{?1??~=f?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?]?V$&??!????? 2@)+~??7e?1?Ga?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 23.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?62.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??0????I???mrU@Qm????K)@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????@?????@!?????@      ??!       "		Q????	Q????!	Q????*      ??!       2	??S??????S????!??S????:	??>Ȳ @??>Ȳ @!??>Ȳ @B      ??!       J	t$???~??t$???~??!t$???~??R      ??!       Z	t$???~??t$???~??!t$???~??b      ??!       JGPUY??0????b q???mrU@ym????K)@