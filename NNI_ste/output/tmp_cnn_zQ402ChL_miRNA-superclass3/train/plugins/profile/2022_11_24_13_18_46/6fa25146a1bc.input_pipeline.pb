	?\??J0?@?\??J0?@!?\??J0?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?\??J0?@?^'?ew@1??E?hg@A?R?????I{?ۡa?9@rEagerKernelExecute 0*	V-???b@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?5^?I??!#?(???@@)=?!7???1???)?@:Preprocessing2F
Iterator::Model>?x????!6r ,??H@)??׻??1/oi??>@:Preprocessing2U
Iterator::Model::ParallelMapV2#??E????!=u<??u2@)#??E????1=u<??u2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeateM.Ɛ?!L:۫u?%@)䃞ͪυ?1?Qn?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??C?X???!ɍ??gxI@)???խ~?1?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorP?,?cyw?!!????C@)P?,?cyw?1!????C@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapi??I??!Z<?l?A@)l=C8f?c?1?ƮPj???:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensorE+??Ba?!Y?? A??)E+??Ba?1Y?? A??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceʩ?ajK]?!:Yy???)ʩ?ajK]?1:Yy???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 63.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?<W<??P@QƆQ?S@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?^'?ew@?^'?ew@!?^'?ew@      ??!       "	??E?hg@??E?hg@!??E?hg@*      ??!       2	?R??????R?????!?R?????:	{?ۡa?9@{?ۡa?9@!{?ۡa?9@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?<W<??P@yƆQ?S@@