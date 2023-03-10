?	d*??G@d*??G@!d*??G@	?2q??@?2q??@!?2q??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLd*??G@??\i??1???cw???A?&N?w(??I????@'@YPQ?+???rEagerKernelExecute 0*	u?V"g@2F
Iterator::Model"m?OT6??!?G?_C?I@)???c"??1?4B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????%ǭ?!?Vy?l?@)b? ??%??1?XԶ?44@:Preprocessing2U
Iterator::Model::ParallelMapV2? 3??O??!ټ????-@)? 3??O??1ټ????-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???s(C??!?w?Op&@)???s(C??1?w?Op&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????????!?x\?p?@)????????1?x\?p?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?h:;??!??G&@)?=ϟ6???1??k???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?{b?*??!6?j??rH@)!#?????1?#???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??? !ʗ?!˥??)@)-y<-?pe?1/???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?39.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t26.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?2q??@I?7?uz?P@Q??2%\?;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??\i????\i??!??\i??      ??!       "	???cw??????cw???!???cw???*      ??!       2	?&N?w(???&N?w(??!?&N?w(??:	????@'@????@'@!????@'@B      ??!       J	PQ?+???PQ?+???!PQ?+???R      ??!       Z	PQ?+???PQ?+???!PQ?+???b      ??!       JGPUY?2q??@b q?7?uz?P@y??2%\?;@?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput|??
T???!|??
T???0"1
model/Conv1D_4/conv1dConv2D??tk????!?;?Ż?"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad0??e??!?U?*???"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??Ҥ
??!????Ө??0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits???bcj??!0???l???"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputʉ?????!Q?T?S!??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??H,???!??{?@??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradx??|Ȉ??!=???Q??"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter7?W?????!???z?Q??0"1
model/Conv1D_1/conv1dConv2D?x?ʨ??!na?C??Q      Y@Y@n]?G*@a8R4??U@q??{&?:@y8	?[???"?
both?Your program is MODERATELY input-bound because 5.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?39.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t26.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?26.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 