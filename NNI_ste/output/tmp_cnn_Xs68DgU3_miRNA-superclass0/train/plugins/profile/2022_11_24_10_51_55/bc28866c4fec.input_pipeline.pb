	Q???`@Q???`@!Q???`@	??XPmz@??XPmz@!??XPmz@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLQ???`@t?????1???[v???AeT?? ??IƤ???@Y??5?e??rEagerKernelExecute 0*	j?t?8i@2U
Iterator::Model::ParallelMapV2ٴR???!9?um`>@)ٴR???19?um`>@:Preprocessing2F
Iterator::Modelݗ3????!?P2?L@)?z?G???1???:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Nw?xΦ?!?}ի6@)?R]????10~??p2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipǃ-v????!c???@?E@),cC7???1???e?W#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??????!Q\ef?@)??????1Q\ef?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate1x??????!p?QJ?I%@)e8?πz??1˫G/??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorT???f~?!?.?A@)T???f~?1?.?A@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$EdX???!??M?N(@)?g^??h?1????|#??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?40.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t19.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??XPmz@I??B[?VN@Q]Zd?Y??@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	t?????t?????!t?????      ??!       "	???[v??????[v???!???[v???*      ??!       2	eT?? ??eT?? ??!eT?? ??:	Ƥ???@Ƥ???@!Ƥ???@B      ??!       J	??5?e????5?e??!??5?e??R      ??!       Z	??5?e????5?e??!??5?e??b      ??!       JGPUY??XPmz@b q??B[?VN@y]Zd?Y??@