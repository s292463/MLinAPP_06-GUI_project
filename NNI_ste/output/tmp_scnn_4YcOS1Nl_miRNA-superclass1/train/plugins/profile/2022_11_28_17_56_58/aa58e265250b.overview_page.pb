?	IZ???8@IZ???8@!IZ???8@	?VD? @?VD? @!?VD? @"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0IZ???8@?oB!???1?Pj/??@I?D??f?4@Yv???;z??r0*	䥛? 2t@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat/???0??!/??Uk)R@)????Z???1???YQ@:Preprocessing2U
Iterator::Model::ParallelMapV2?%????!?0?"$@)?%????1?0?"$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???g%???!E???($@)o??ʡ??1?Eu&@:Preprocessing2F
Iterator::Model5(??ȧ?!^4%?j?,@)V(??????1???C?;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceY1\ q??!?	?,jV@)Y1\ q??1?	?,jV@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorO??Z}??!b?Y?h?	@)O??Z}??1b?Y?h?	@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipD? ???!uY???gU@)K??^b,??1?w?#?-@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?83.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?VD? @I??vU@Q????[%@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?oB!????oB!???!?oB!???      ??!       "	?Pj/??@?Pj/??@!?Pj/??@*      ??!       2      ??!       :	?D??f?4@?D??f?4@!?D??f?4@B      ??!       J	v???;z??v???;z??!v???;z??R      ??!       Z	v???;z??v???;z??!v???;z??b      ??!       JGPUY?VD? @b q??vU@y????[%@?".
IteratorGetNext/_29_Send`?FM@???!`?FM@???"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInputʕ???1??!?n?????0"?
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolutionConv2DO2??????!5??o??"?
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolutionConv2D?J4???!?۱????"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterl+?D??!???;c??0"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput}?P?Q??!?	fM???0"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_6/convolution_grad/Conv2DBackpropInputConv2DBackpropInput????p??!y賐W???0"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/SparseDotIncBuilder/SparseTensorDenseMatMul_grad/SparseTensorDenseMatMulSparseTensorDenseMatMulxig???!`y:???"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput_5?SF???!??wfr???0"?
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_5/convolutionConv2D????f??!?#\????Q      Y@Y??>r?)@a?'????U@q?????@yƆu???"?

device?Your program is NOT input-bound because only 3.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?83.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 