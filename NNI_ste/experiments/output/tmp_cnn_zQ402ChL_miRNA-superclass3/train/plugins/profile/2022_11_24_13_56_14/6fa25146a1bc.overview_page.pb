?	׾?^?? @׾?^?? @!׾?^?? @      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC׾?^?? @߉Y/?@1???x????A??bc^G??I]?@??@rEagerKernelExecute 0*	1?Z?c@2F
Iterator::ModelA??4F???!?ʃ@Y J@)??????1?P4=??B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???1ZG??!?Z??r:@)???WW??1tW)f6@:Preprocessing2U
Iterator::Model::ParallelMapV2??s??Ɨ?!|?=ԍ-@)??s??Ɨ?1|?=ԍ-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice8????C??!^1b??f@)8????C??1^1b??f@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipc~nh?N??!5|???G@)C=}????18???B?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatez ???!??!RD???+@)??X ??1?r&a%?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorw?x?z?!?	
?2@)w?x?z?1?	
?2@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapv?1<????!??2}a?.@)??6?ُd?1d?rﶎ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 29.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?50.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI|B?p\%T@Q??=?j3@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	߉Y/?@߉Y/?@!߉Y/?@      ??!       "	???x???????x????!???x????*      ??!       2	??bc^G????bc^G??!??bc^G??:	]?@??@]?@??@!]?@??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q|B?p\%T@y??=?j3@?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputlL?h???!lL?h???0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterh??-????!???????0"1
model/Conv1D_2/conv1dConv2DB??p???!۶xG???"1
model/Conv1D_3/conv1dConv2D??l:???!??i?
[??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter{?MJ????!????h???0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradXd? /??!?s??L???"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputk\{F#??!?2õc??0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad@?<?Ӡ?!Z?z?/~??"1
model/Conv1D_1/conv1dConv2D/?m>U???!?ga?Tm??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???????!?q,?\??0Q      Y@Y!Y?B*@a????7?U@q??Cǎ3X@yM~?????"?
both?Your program is POTENTIALLY input-bound because 29.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?50.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 