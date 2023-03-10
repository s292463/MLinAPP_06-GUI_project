?	??:M?@??:M?@!??:M?@	?!#	?????!#	????!?!#	????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??:M?@<L?????1?}?֤??Az ???!??I??-??@Y_9?????rEagerKernelExecute 0*	?$???a@2F
Iterator::ModelQJV?˳?!?&uQH?J@)ϼv?1??1???#C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????٢?!?G?p??9@)??7h?>??1&?????4@:Preprocessing2U
Iterator::Model::ParallelMapV2???k?˖?!?}?^??.@)???k?˖?1?}?^??.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???RAE??!)E??@)???RAE??1)E??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?2???V??!???A*@)??cx?g??1f!?⇡@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??N??!zي??G@)? !????1?3W?b?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?o????}?!ȵ?FG?@)?o????}?1ȵ?FG?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap!??????!?<?y??-@)j?????e?1?X/W???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 23.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?57.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?!#	????I??o?l\T@QϮy??0@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	<L?????<L?????!<L?????      ??!       "	?}?֤???}?֤??!?}?֤??*      ??!       2	z ???!??z ???!??!z ???!??:	??-??@??-??@!??-??@B      ??!       J	_9?????_9?????!_9?????R      ??!       Z	_9?????_9?????!_9?????b      ??!       JGPUY?!#	????b q??o?l\T@yϮy??0@?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??Z!????!??Z!????0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?d?Hu??!}36?2??0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits$??&(??!????????"1
model/Conv1D_2/conv1dConv2D:?r?#,??!`??0????"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterRD[?aR??!:9???0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad3??"???!?x?"???"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad??7l?Ţ?!?nF????"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad?}?+<??!o>?Qa??"1
model/Conv1D_3/conv1dConv2D?A?????!AZ>?:??"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput??I@?[??!??B ????0Q      Y@Y     ?-@a     JU@qY??݆G@y-0a???"?
both?Your program is POTENTIALLY input-bound because 23.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?57.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?47.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 