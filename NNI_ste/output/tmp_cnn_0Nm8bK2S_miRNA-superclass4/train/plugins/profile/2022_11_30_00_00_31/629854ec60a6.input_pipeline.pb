	?h????)@?h????)@!?h????)@	3??/ܢ@3??/ܢ@!3??/ܢ@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?h????)@Ȗ??2?@1?H???@A???{???I???{?@YsI?v|??rEagerKernelExecute 0*??? ??g@)       =2F
Iterator::Model??????!dX0???G@)D?3?<??1??q}|?>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX?|[?T??!?::???7@)U??7???1?d V??4@:Preprocessing2U
Iterator::Model::ParallelMapV2? ?M?ܟ?!H*???:0@)? ?M?ܟ?1H*???:0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceA???FX??!z??,?$@)A???FX??1z??,?$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipu<f?2???!???[F{J@);8؛???1p?i?D?#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???GS=??!ӓH?1@)z??L?D??1V ?<??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?uʣ{?!#????(@)?uʣ{?1#????(@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapt^c??ޢ?!?)??u93@)R???Tj?1y^i	ؒ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 23.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?17.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no94??/ܢ@I?/??իD@Q?2"???K@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ȗ??2?@Ȗ??2?@!Ȗ??2?@      ??!       "	?H???@?H???@!?H???@*      ??!       2	???{??????{???!???{???:	???{?@???{?@!???{?@B      ??!       J	sI?v|??sI?v|??!sI?v|??R      ??!       Z	sI?v|??sI?v|??!sI?v|??b      ??!       JGPUY4??/ܢ@b q?/??իD@y?2"???K@