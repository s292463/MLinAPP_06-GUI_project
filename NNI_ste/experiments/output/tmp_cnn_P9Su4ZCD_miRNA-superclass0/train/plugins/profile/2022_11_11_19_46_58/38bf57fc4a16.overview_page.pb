?	B@??Z!@B@??Z!@!B@??Z!@	l??ا???l??ا???!l??ا???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLB@??Z!@7??? @12 {?????Ar?z?f???I??+f@Y??ĭ???rEagerKernelExecute 0*	????M
`@2F
Iterator::Modelծ	i?A??!7?l?a?G@)??~????1?_HT?<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?KS8???!Ɵo@>@)"q??]??1??????8@:Preprocessing2U
Iterator::Model::ParallelMapV2׽?	j??!?*?Ao?2@)׽?	j??1?*?Ao?2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????H??!Ǯd??2 @)?????H??1Ǯd??2 @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?DkE???!?l????-@)?h??????1?{?t?I@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipb??4?8??!?:?\?6J@)մ?i?{}?1n??K?o@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???B{?!6?5*?@)???B{?16?5*?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap? Q0c
??!p? w??0@)W'g(?xc?1.1?+P???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 23.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?52.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9l??ا???I7????S@Q????6@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	7??? @7??? @!7??? @      ??!       "	2 {?????2 {?????!2 {?????*      ??!       2	r?z?f???r?z?f???!r?z?f???:	??+f@??+f@!??+f@B      ??!       J	??ĭ?????ĭ???!??ĭ???R      ??!       Z	??ĭ?????ĭ???!??ĭ???b      ??!       JGPUYl??ا???b q7????S@y????6@?"1
model/Conv1D_2/conv1dConv2D`??;??!`??;??"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?[+??C??!P??ꃿ??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput????????!??S),??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?N38?l??!?ÓP????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad????nR??!*??/????"1
model/Conv1D_3/conv1dConv2D$?X???!???FJ??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad?l!8*??!H?؍/??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput??#w???!̴ W ??0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits???i?~??!?]??B???"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose-!?i\8??!?oM]?[??Q      Y@Y,Q??+'@aە?]?V@qf????tD@y1?4?????"?
both?Your program is POTENTIALLY input-bound because 23.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?52.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?40.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 