	eo)??]@eo)??]@!eo)??]@	???cD?@???cD?@!???cD?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLeo)??]@??_????1Ѱu?@A??a?? ??I?Sͬe@Y?s??q5??rEagerKernelExecute 0*	V-??a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?S?K???!?0Q?z@@)??
???1ԍ(?n?;@:Preprocessing2F
Iterator::Model1????4??!?q??cA@)?2?Pl??1wit4@:Preprocessing2U
Iterator::Model::ParallelMapV2?sb?c??!.>????-@)?sb?c??1.>????-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceȶ8Kɒ?!*ss???)@)ȶ8Kɒ?1*ss???)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateW?'???!?*G??5@)???????1??????!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?)??z???!wG,-NP@)'?E'K???1+C???U @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??@J??~?!?
??fI@)??@J??~?1?
??fI@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapD?1uWv??!Ƿ??8@)??u?Tj?1?e f
*@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 17.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?39.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???cD?@I内$??L@Q?=??}C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??_??????_????!??_????      ??!       "	Ѱu?@Ѱu?@!Ѱu?@*      ??!       2	??a?? ????a?? ??!??a?? ??:	?Sͬe@?Sͬe@!?Sͬe@B      ??!       J	?s??q5???s??q5??!?s??q5??R      ??!       Z	?s??q5???s??q5??!?s??q5??b      ??!       JGPUY???cD?@b q内$??L@y?=??}C@