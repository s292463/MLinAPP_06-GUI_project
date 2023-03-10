?	LnYk.@LnYk.@!LnYk.@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCLnYk.@3??????1y?n?|J'@A????Fu??I9??cx???rEagerKernelExecute 0*	?A`??d@2F
Iterator::ModelJ^?c@???!B??S?K@)eq???б?1<?E`"?E@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat()? ???!?k?l9@)?}9?]??1??]5@:Preprocessing2U
Iterator::Model::ParallelMapV2??wF[???!L ??)@)??wF[???1L ??)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?ZC?????!??Q?"h@)?ZC?????1??Q?"h@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?{?i????!??Yl?G(@)М?)?d??1??a?'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???,&??!??q|?F@)1???z??1???9@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor@?Z?kBz?!?w?@)@?Z?kBz?1?w?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap]?jJ???!		M?,@)?\p?h?1?:?Gq??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 10.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?12.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI? ]o7@Q???>$S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	3??????3??????!3??????      ??!       "	y?n?|J'@y?n?|J'@!y?n?|J'@*      ??!       2	????Fu??????Fu??!????Fu??:	9??cx???9??cx???!9??cx???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q? ]o7@y???>$S@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterw??|g!??!w??|g!??0"1
model/Conv1D_2/conv1dConv2D????????!?Z??a??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterBx????!?=??}??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?????٪?!??׮?3??0"1
model/Conv1D_3/conv1dConv2D??|?????!~{'U???"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?`?[.???!$???Z???0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradk"???ݢ?!q.????"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad&<?$???!3?????"3
model/Conv1D_1/BiasAddBiasAdd?Nb۸a??!ǅ?????"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?_K?:??!??;W???Q      Y@Y.>9\&@a<???x4V@q?51??9@yb???M??"?
both?Your program is POTENTIALLY input-bound because 10.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?12.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?25.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 