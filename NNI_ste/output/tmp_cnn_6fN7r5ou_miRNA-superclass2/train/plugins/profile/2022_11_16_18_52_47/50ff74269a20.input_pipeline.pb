	??̯?tF@??̯?tF@!??̯?tF@	????????????????!????????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??̯?tF@kb??????1?y??w?@A??!????I=?e?YXC@YW\?????rEagerKernelExecute 0*	?v???d@2F
Iterator::Model?8*7QK??!j"?B"?K@)?(??=$??1E~>\'C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat^H??0~??!Z?$c??5@)*?#??t??16AX?2@:Preprocessing2U
Iterator::Model::ParallelMapV2?> ?M???!K^??0@)?> ?M???1K^??0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip ??4?ײ?!??R??[F@)-|}?K???1r1???#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??a?????!?®D??@)??a?????1?®D??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????a???!w??B?&@)?M?????1
??+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorH2?w?z?!I*WX?@)H2?w?z?1I*WX?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???,Օ?!0?c+?)@) ??*Q?f?1?͸?E???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?86.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9????????ION??UV@QF?????$@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	kb??????kb??????!kb??????      ??!       "	?y??w?@?y??w?@!?y??w?@*      ??!       2	??!??????!????!??!????:	=?e?YXC@=?e?YXC@!=?e?YXC@B      ??!       J	W\?????W\?????!W\?????R      ??!       Z	W\?????W\?????!W\?????b      ??!       JGPUY????????b qON??UV@yF?????$@