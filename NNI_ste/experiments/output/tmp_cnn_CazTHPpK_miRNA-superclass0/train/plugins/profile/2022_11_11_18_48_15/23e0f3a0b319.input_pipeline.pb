	??0?!@??0?!@!??0?!@	???b_?????b_??!???b_??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??0?!@jL??????16?ڋh???Ad??3?Ĩ?I??q?j?@Y???A_z??rEagerKernelExecute 0*	?G?zd`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat8,?????!xw|??@@)??y?):??1?lU?%;@:Preprocessing2F
Iterator::Model3??(???!??Ԯ?	G@)n??S??1?~????:@:Preprocessing2U
Iterator::Model::ParallelMapV2\??????!?&?B3@)\??????1?&?B3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???i???!??˶?	 @)???i???1??˶?	 @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?N?`????!???<r:-@)?%!????1?e?a@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipT?^P??!B8+Q?J@)??G??5|?1???ƭ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor,???cz?!#	????@),???cz?1#	????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??M~?N??!g??7{?0@)??$?pte?1?g??!???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?64.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???b_??IB?P??,U@Q?a"?o,,@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	jL??????jL??????!jL??????      ??!       "	6?ڋh???6?ڋh???!6?ڋh???*      ??!       2	d??3?Ĩ?d??3?Ĩ?!d??3?Ĩ?:	??q?j?@??q?j?@!??q?j?@B      ??!       J	???A_z?????A_z??!???A_z??R      ??!       Z	???A_z?????A_z??!???A_z??b      ??!       JGPUY???b_??b qB?P??,U@y?a"?o,,@