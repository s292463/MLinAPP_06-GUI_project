	 9a?h?C@ 9a?h?C@! 9a?h?C@	???0?P?????0?P??!???0?P??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL 9a?h?C@??R??1??1Pō[L@A?t?_?ʔ?I`:?۠?@@Y<?_?E??rEagerKernelExecute 0*	=
ףp?f@2F
Iterator::Model?=]ݱز?!???
D@)u?׃I???1+2?=a?;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?c?1??!%6??@@)W@??>??10?ܣw3@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??;????!4?V??~)@)??;????14?V??~)@:Preprocessing2U
Iterator::Model::ParallelMapV29??m4???!r?K?g)@)9??m4???1r?K?g)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_a?????!????u?'@)_a?????1????u?'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip>U?W??!?ot??M@)?wG?j???1?] ?!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate`?U,~S??!???4?k1@)?bԵ?>??1??HSz?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????ء?!??.??
3@)??ϛ?Th?1,???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?85.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???0?P??I???q5V@Q?ƽ`e?$@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??R??1????R??1??!??R??1??      ??!       "	Pō[L@Pō[L@!Pō[L@*      ??!       2	?t?_?ʔ??t?_?ʔ?!?t?_?ʔ?:	`:?۠?@@`:?۠?@@!`:?۠?@@B      ??!       J	<?_?E??<?_?E??!<?_?E??R      ??!       Z	<?_?E??<?_?E??!<?_?E??b      ??!       JGPUY???0?P??b q???q5V@y?ƽ`e?$@