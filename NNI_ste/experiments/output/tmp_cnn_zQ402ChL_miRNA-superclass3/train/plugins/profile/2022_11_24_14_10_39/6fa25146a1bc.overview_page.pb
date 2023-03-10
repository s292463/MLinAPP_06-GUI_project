?	??-??@??-??@!??-??@	?H)?????H)????!?H)????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??-??@q?::?f @1?eN??? @A??W?ۜ?I?)t^c?
@Y???eN???rEagerKernelExecute 0*	????K?b@2F
Iterator::Model?jIG9??!?Ns???J@)
?y???1?I[??L@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?????Ԥ?!?G??X	;@)???l亡?1#?N>7@:Preprocessing2U
Iterator::Model::ParallelMapV2`??橞?!P	0a?3@)`??橞?1P	0a?3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceh?
?O??!޹ے??@)h?
?O??1޹ے??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateO?)??Y??!	?-j*@)6sHj?d??15Jɍ?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??Y?rL??!h??v*?G@)1{?v???1?Rr@?3@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorZ??? ?x?!%??j@)Z??? ?x?1%??j@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?<L???!???5?-@)?5x_?e?1?VЇ?P??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 26.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?43.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?H)????IP???`?Q@Q1???H;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	q?::?f @q?::?f @!q?::?f @      ??!       "	?eN??? @?eN??? @!?eN??? @*      ??!       2	??W?ۜ???W?ۜ?!??W?ۜ?:	?)t^c?
@?)t^c?
@!?)t^c?
@B      ??!       J	???eN??????eN???!???eN???R      ??!       Z	???eN??????eN???!???eN???b      ??!       JGPUY?H)????b qP???`?Q@y1???H;@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterG?<f??!G?<f??0"1
model/Conv1D_2/conv1dConv2D4?????!)?L???"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??l????!'Q?S???0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad$M???A??!8 ?????"1
model/Conv1D_1/conv1dConv2D[^V????!??؈A??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???n?8??!?S?F????0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsA`?6?V??!??/Z^??"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput?????I??!/e
e????0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput??L?=??!?d??????0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad/4W????!}?>????Q      Y@Yyxxxxx*@a??????U@qG?9??iD@yL[7vP??"?
both?Your program is POTENTIALLY input-bound because 26.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?43.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?40.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 