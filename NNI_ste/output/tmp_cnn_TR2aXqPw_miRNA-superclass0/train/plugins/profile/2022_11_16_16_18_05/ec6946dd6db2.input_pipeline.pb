	~????@~????@!~????@	?l?)@?l?)@!?l?)@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL~????@"??p???1??'??@Au?_????I"????@YZ??!? ??rEagerKernelExecute 0*	????Mvr@2U
Iterator::Model::ParallelMapV2? ??n??!???r	K@)? ??n??1???r	K@:Preprocessing2F
Iterator::Modela??*??!?T???Q@)???%ǝ??1??c99?1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?G?&??![???J1@)MJA??4??1)??S:]-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????1??!o?$jY@)????1??1o?$jY@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?(?r??!?.??\<@)?NGɫ??1?xNA
@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?????ޔ?!?O?Y0?@)???͋??1??,???	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?[?~l??!T? &?@)?[?~l??1T? &?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??n????!????F@)?D?e??f?1???O"l??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?34.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?l?)@I?j???uI@Q?>=?7G@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	"??p???"??p???!"??p???      ??!       "	??'??@??'??@!??'??@*      ??!       2	u?_????u?_????!u?_????:	"????@"????@!"????@B      ??!       J	Z??!? ??Z??!? ??!Z??!? ??R      ??!       Z	Z??!? ??Z??!? ??!Z??!? ??b      ??!       JGPUY?l?)@b q?j???uI@y?>=?7G@