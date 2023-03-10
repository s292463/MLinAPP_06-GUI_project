?	ׅ?O@ׅ?O@!ׅ?O@	???4)~@???4)~@!???4)~@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLׅ?O@?I??????1??T?z??A?-?s`??IM???x	@Y.V?`???rEagerKernelExecute 0*	Zd;߫d@2F
Iterator::Model˾+?????!B????C@)>????1y??2??8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?	?Y2??!??rU?e;@)???W;???1鮬&?7@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???5w???!T?f???2@)???5w???1T?f???2@:Preprocessing2U
Iterator::Model::ParallelMapV2$0??{??!?'Z??,@)$0??{??1?'Z??,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?V???x??!??T?%\9@)V*?????1j-??f?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip3?z????!??LN@)z?΅?^??1I?	<?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor8??@}?!?n?lF@)8??@}?1?n?lF@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapߣ?z???!?????0;@)>?4a??h?1&?b??F??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?43.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t23.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???4)~@I???z?P@Q???+,:@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?I???????I??????!?I??????      ??!       "	??T?z????T?z??!??T?z??*      ??!       2	?-?s`???-?s`??!?-?s`??:	M???x	@M???x	@!M???x	@B      ??!       J	.V?`???.V?`???!.V?`???R      ??!       Z	.V?`???.V?`???!.V?`???b      ??!       JGPUY???4)~@b q???z?P@y???+,:@?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilters(???.??!s(???.??0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput???Ew???!9x֘????0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad?<???]??!Z?R	???"1
model/Conv1D_4/conv1dConv2D????:9??!B??W???"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput?h?b??!@N?7????0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?Z<HTQ??!?????Y??0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsP$?????!~??ۻ??"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterG???6???!?H?C????0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??@??!H?g??^??0"1
model/Conv1D_3/conv1dConv2D??؉???!ņ????Q      Y@Yݘ??V+@a`?.?U@q?_??n<@y???CL???"?
both?Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?43.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t23.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?28.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 