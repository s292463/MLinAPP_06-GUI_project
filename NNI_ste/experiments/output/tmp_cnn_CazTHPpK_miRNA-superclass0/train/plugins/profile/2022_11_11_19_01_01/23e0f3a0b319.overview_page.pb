?	??o^??"@??o^??"@!??o^??"@	??$r?@??$r?@!??$r?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??o^??"@|~!???1??8՚??A??L?ϫ?Im????@Y?Z?7????rEagerKernelExecute 0*	??"???a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ek}???!?0.'b?@@)?C?Ö??1?	?=@:Preprocessing2F
Iterator::Model`X?|[???!<D? ?C@)?/ע??1?2???
6@:Preprocessing2U
Iterator::Model::ParallelMapV2???KqU??!}U%?,m1@)???KqU??1}U%?,m1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??+d???! ?Q;8?1@)d??1ˎ?1??*?.%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?>+N??!3RG??O@)?>+N??13RG??O@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip9??????!Ļt??CN@)??̔?߂?1??f?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?/K;5?{?!?Jњ?@)?/K;5?{?1?Jњ?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?"rl??!?c?a=4@)?????j?1䐒?K?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?55.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t19.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??$r?@Iz5?u?R@Q?&?61@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	|~!???|~!???!|~!???      ??!       "	??8՚????8՚??!??8՚??*      ??!       2	??L?ϫ???L?ϫ?!??L?ϫ?:	m????@m????@!m????@B      ??!       J	?Z?7?????Z?7????!?Z?7????R      ??!       Z	?Z?7?????Z?7????!?Z?7????b      ??!       JGPUY??$r?@b qz5?u?R@y?&?61@?"1
model/Conv1D_2/conv1dConv2Dd?ҵ????!d?ҵ????"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???F?H??!],?	???0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?Ւ?????!v?- p??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??4T?`??!+ O????0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits??3?Y???!??ռ????"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad????}??!8?y?wG??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?	Y%???!??Kh??0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput?+O????!3??0???0"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad9{???!?5Bn?N??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad??:0???!?E!g???Q      Y@Y     ?'@a     V@q$N?G??4@y????y???"?
both?Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?55.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t19.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?20.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 