	j.7?@j.7?@!j.7?@	o??N???o??N???!o??N???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLj.7?@???????1?'?8g@AmU?Y??Ir?&" 	@Y??i????rEagerKernelExecute 0*	F????0d@2F
Iterator::Model??ڦx\??!wqĞH@)C8???1???['B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate _B???!? ?X??@@)?&???K??1?K????@:Preprocessing2U
Iterator::Model::ParallelMapV2j?@+0d??!??c??)@)j?@+0d??1??c??)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?:pΈ??!{? ?Gi&@)?aod??19?q_@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Z??K???!????;aI@)~???|?1v??1?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?*??p?w?!?JB=?@)?*??p?w?1?JB=?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????m3??!N????A@)?h8en?a?1???t??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor߿yq??]?!???@\???)߿yq??]?1???@\???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicerQ-"??[?!G?~޽???)rQ-"??[?1G?~޽???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?46.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9n??N???I??"??!P@Q?`D?q?@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????????????!???????      ??!       "	?'?8g@?'?8g@!?'?8g@*      ??!       2	mU?Y??mU?Y??!mU?Y??:	r?&" 	@r?&" 	@!r?&" 	@B      ??!       J	??i??????i????!??i????R      ??!       Z	??i??????i????!??i????b      ??!       JGPUYn??N???b q??"??!P@y?`D?q?@@