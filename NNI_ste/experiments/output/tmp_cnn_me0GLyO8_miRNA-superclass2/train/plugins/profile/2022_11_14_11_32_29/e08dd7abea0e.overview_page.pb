?	B%?c\?@B%?c\?@!B%?c\?@	Õ?m8@Õ?m8@!Õ?m8@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLB%?c\?@??I}Y???1??)"C??A*?:]???I1[?*?@YZ??լ3??rEagerKernelExecute 0*	? ?rh?e@2F
Iterator::ModelWZF?=???!Y%A%F@) Sh???1??+O>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat0*??D??!M8?<(U;@)M??E;??1G?$??6@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??h8e??!?????1@)??h8e??1?????1@:Preprocessing2U
Iterator::Model::ParallelMapV2]??'???!D?47>?+@)]??'???1D?47>?+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?d?pu ??!?#|???6@)?`?d7??1??s?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?'?8'??!??a1@)?'?8'??1??a1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??:?ϸ?!?ھ???K@)0?x??n?1?P??R?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap|??l;m??!?h?߸!8@).8??_?f?1nMT?1???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?51.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Õ?m8@I?VSsR@Q??????6@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??I}Y?????I}Y???!??I}Y???      ??!       "	??)"C????)"C??!??)"C??*      ??!       2	*?:]???*?:]???!*?:]???:	1[?*?@1[?*?@!1[?*?@B      ??!       J	Z??լ3??Z??լ3??!Z??լ3??R      ??!       Z	Z??լ3??Z??լ3??!Z??լ3??b      ??!       JGPUYÕ?m8@b q?VSsR@y??????6@?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?rL?Ϩ?!?rL?Ϩ?0"K
$Adam/Adam/update_8/ResourceApplyAdamResourceApplyAdam1??n???!?Cǒ?7??"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput??,???!???M`???0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?o???١?!?A?
??0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits"??z???!??=?1x??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput1???_۞?!vd6??S??0"1
model/Conv1D_4/conv1dConv2D???dڝ?!	??u???"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?WS?u??!?.g?^??"2
model/Dense_1/MatMulMatMul:<CUS??!2?a?$??0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad???!??!0?A????Q      Y@Y??????+@a??????U@q?[nF??A@y㶜^??"?
both?Your program is POTENTIALLY input-bound because 21.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?51.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?35.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 