?	H6W?s?@H6W?s?@!H6W?s?@	w?Qe??@w?Qe??@!w?Qe??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLH6W?s?@>?٬?\??1)狽??A|??c?M??I?Z??C@YZ??/-???rEagerKernelExecute 0*	????ҵa@2F
Iterator::Model7e??!?7????G@)?v????1??	?Ͳ?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat8?{?5Z??!?lf9?>@)??r0? ??1kD??1:@:Preprocessing2U
Iterator::Model::ParallelMapV2?'d?ml??!,?4?V?.@)?'d?ml??1,?4?V?.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice5?8EGr??!?2!?!@)5?8EGr??1?2!?!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?x?????!6???D?-@)*??????1??d/m?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?xZ~?*??!
?m}ClJ@)_??W?{?1?7^h'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(?H0??z?!Z?*?x@)(?H0??z?1Z?*?x@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapB?v????!??]?@1@)??c${?j?1??Z?G@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 24.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?51.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9v?Qe??@I
????S@QgSv #4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	>?٬?\??>?٬?\??!>?٬?\??      ??!       "	)狽??)狽??!)狽??*      ??!       2	|??c?M??|??c?M??!|??c?M??:	?Z??C@?Z??C@!?Z??C@B      ??!       J	Z??/-???Z??/-???!Z??/-???R      ??!       Z	Z??/-???Z??/-???!Z??/-???b      ??!       JGPUYv?Qe??@b q
????S@ygSv #4@?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?iQL50??!?iQL50??0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter7he??f??!??????0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGradJW??]	??!???0???"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput2???????!??]????0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput??xD???!v'?{????0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits=M???!cԆ^?C??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?2????!??I?<F??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad??d?LX??!???????"1
model/Conv1D_2/conv1dConv2D?M?!ќ?!??`?????"1
model/Conv1D_4/conv1dConv2D&F':y??!?ke???Q      Y@Y??????+@a??????U@q??????>@y???OW??"?
both?Your program is POTENTIALLY input-bound because 24.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?51.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?30.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 