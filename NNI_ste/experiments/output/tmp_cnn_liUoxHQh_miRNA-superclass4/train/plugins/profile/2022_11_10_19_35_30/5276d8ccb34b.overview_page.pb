?	????? @????? @!????? @	?????@?????@!?????@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL????? @d?6??? @1??4??@A???|y??II?,|}?@Y?n?????rEagerKernelExecute 0*	?????0`@2F
Iterator::Modelz4Փ?G??!??P???G@)֩?=#??1???N>;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???=?>??!O????>@)a???)??1????9@:Preprocessing2U
Iterator::Model::ParallelMapV2Hū?m??!????3@)Hū?m??1????3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice0??e??!???^?? @)0??e??1???^?? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?
??捓?!ƌK?w|-@)O=?බ??1??h?3@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip{????!?HUjJ@)???N~?1f.?Q??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorcb?qm?x?!?z?$L?@)cb?qm?x?1?z?$L?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Udt@??!L?;??0@)\Y???"d?1??`??\??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 24.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?45.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?????@IL?_$??Q@Q@??? 8:@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	d?6??? @d?6??? @!d?6??? @      ??!       "	??4??@??4??@!??4??@*      ??!       2	???|y?????|y??!???|y??:	I?,|}?@I?,|}?@!I?,|}?@B      ??!       J	?n??????n?????!?n?????R      ??!       Z	?n??????n?????!?n?????b      ??!       JGPUY?????@b qL?_$??Q@y@??? 8:@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFiltery?O?̵?!y?O?̵?0"1
model/Conv1D_2/conv1dConv2DV?xheX??!hgd?8???"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?aH]??!4$?????0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterz?']~??!?/	:???0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputJf?YҠ?!??U6????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad螝???!ɬ?vv???"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad????	???!??5????"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradЬ<1+a??!e?IŹK??"1
model/Conv1D_3/conv1dConv2D??n2Ì??!0yp?????"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?S???}??!k.|sa???0Q      Y@YH?R&?&@a??5?'V@q?????E@y[??b]???"?
both?Your program is POTENTIALLY input-bound because 24.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?45.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?43.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 