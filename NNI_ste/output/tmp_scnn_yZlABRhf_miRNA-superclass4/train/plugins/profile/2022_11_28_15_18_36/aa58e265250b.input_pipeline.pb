	??6T?H@??6T?H@!??6T?H@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??6T?H@nk?K??1h?N??	@@I?^D?1=/@r0*	??Q?qg@2F
Iterator::Model)??????!u6?ukB@)??????1u??1?6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat]Ot]????!?? ?o?:@)@Û5x_??1Z??>bB6@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??F??R??!??8???O@)i???!??1?Ax?L-@:Preprocessing2U
Iterator::Model::ParallelMapV2??@gҦ??!??͛??+@)??@gҦ??1??͛??+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice%<?ןė?!bj
???(@)%<?ןė?1bj
???(@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?NҤ?!TPd]?5@)??yT?ߑ?1G??Ý"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorw0b? ???!??o?6D@)w0b? ???1??o?6D@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?32.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI????C?@@Q???&ްP@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	nk?K??nk?K??!nk?K??      ??!       "	h?N??	@@h?N??	@@!h?N??	@@*      ??!       2      ??!       :	?^D?1=/@?^D?1=/@!?^D?1=/@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????C?@@y???&ްP@