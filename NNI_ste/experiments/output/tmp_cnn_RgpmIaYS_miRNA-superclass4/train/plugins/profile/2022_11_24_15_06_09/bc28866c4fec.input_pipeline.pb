	? d?*$@? d?*$@!? d?*$@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC? d?*$@??-Y???1??M??p@A?ܚt["??IB@??*@rEagerKernelExecute 0*	o???mg@2F
Iterator::Modelh??`ob??!?VD+?hI@)yZ~?*O??1/??޻?@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??tީ?!?L?y?:@)?????g??1?k'??X7@:Preprocessing2U
Iterator::Model::ParallelMapV2?!? ?&??!????d?0@)?!? ?&??1????d?0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateu??<???!?Z???0@) ?߽?Ɣ?1?=?J?%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?6???N??!@?@?s)@)?6???N??1@?@?s)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??`?
???!????H@)??ם?<??1?H??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorKW??x?{?!?	?O??@)KW??x?{?1?	?O??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapi??Q???!?|?	?1@)???7??h?1??i????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?22.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??x?D@Q?Q*??M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??-Y?????-Y???!??-Y???      ??!       "	??M??p@??M??p@!??M??p@*      ??!       2	?ܚt["???ܚt["??!?ܚt["??:	B@??*@B@??*@!B@??*@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??x?D@y?Q*??M@