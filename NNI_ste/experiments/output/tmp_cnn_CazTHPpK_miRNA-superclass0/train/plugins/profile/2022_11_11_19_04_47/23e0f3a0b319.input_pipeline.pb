	N?@?C?!@N?@?C?!@!N?@?C?!@	???/??????/???!???/???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLN?@?C?!@?wG?jS @1???,???A?n?|?b??I???x??@Y#??u???rEagerKernelExecute 0*	?$??3`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX9??v??!;?u???@@)???????1{I?bz<@:Preprocessing2F
Iterator::Model?~l????!ې???E@)w?$$?6??1I?mZ??6@:Preprocessing2U
Iterator::Model::ParallelMapV2?Z? m???!mD	?>W3@)?Z? m???1mD	?>W3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice:?6U??!??????!@):?6U??1??????!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?B??˔?!?
ِV/@)?JC?B??1'Ϝ??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip4???5??!&oDny?L@)=??@fg??1ԫ???9@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor"P??H?|?!??
?}@)"P??H?|?1??
?}@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap,g~5??!?Ts^?|1@)?Ia??Lc?1?o??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 23.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?58.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???/???I~U(RفT@Q?^??I0@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?wG?jS @?wG?jS @!?wG?jS @      ??!       "	???,??????,???!???,???*      ??!       2	?n?|?b???n?|?b??!?n?|?b??:	???x??@???x??@!???x??@B      ??!       J	#??u???#??u???!#??u???R      ??!       Z	#??u???#??u???!#??u???b      ??!       JGPUY???/???b q~U(RفT@y?^??I0@