	?}???%@?}???%@!?}???%@	m;JD?b
@m;JD?b
@!m;JD?b
@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?}???%@?a???R??1)!XU/?@AS#?3????I#??^L??Y/l?V^???rEagerKernelExecute 0*	&??C?|@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?7?k????!Q??^:O@)S?{/???1g???M@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?8~?4??!E???h?.@)R?Q???1?^? ?*@:Preprocessing2F
Iterator::Modelx??e??!?m!??2@)m????U??1??ӌ'@:Preprocessing2U
Iterator::Model::ParallelMapV2?E???Ԡ?!ާ???]@)?E???Ԡ?1ާ???]@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipK?R??%??!???7YT@)??
???1?ط??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?(??0??!??U?9@)?(??0??1??U?9@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???i????!D?Ĭ????)???i????1D?Ĭ????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap? l@????! ?쟌O@)j?drjgh?1?U?@???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?17.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9n;JD?b
@I?(??_@@Q8???O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?a???R???a???R??!?a???R??      ??!       "	)!XU/?@)!XU/?@!)!XU/?@*      ??!       2	S#?3????S#?3????!S#?3????:	#??^L??#??^L??!#??^L??B      ??!       J	/l?V^???/l?V^???!/l?V^???R      ??!       Z	/l?V^???/l?V^???!/l?V^???b      ??!       JGPUYn;JD?b
@b q?(??_@@y8???O@