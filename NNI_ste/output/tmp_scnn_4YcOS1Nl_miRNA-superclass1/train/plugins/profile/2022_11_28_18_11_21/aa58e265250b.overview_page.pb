?	????	3@????	3@!????	3@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'????	3@??Z?-@1?y????I1?74e?@r0*	?Zd;?x@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapv??fG???!?9??~K@)?}q?J[??1ҟ?he4H@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??Im ??!c?9ŭ?1@)8??+خ?1????.@:Preprocessing2U
Iterator::Model::ParallelMapV2'ݖ?g??!^I??@(@)'ݖ?g??1^I??@(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip=b??B??!H????"T@)?????ٞ?1j?i???@:Preprocessing2F
Iterator::ModelWZF?=???!?f?,?u3@)??솝?1??W@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice%???w??!???[?M@)%???w??1???[?M@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???ӹ???!%e?Ł@)???ӹ???1%e?Ł@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 77.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?18.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??@???W@Q?!?+eg@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??Z?-@??Z?-@!??Z?-@      ??!       "	?y?????y????!?y????*      ??!       2      ??!       :	1?74e?@1?74e?@!1?74e?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??@???W@y?!?+eg@?".
IteratorGetNext/_30_Recv???M????!???M????"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInputxN)A???!p? ?C???0"?
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolutionConv2D?r6??Y??!?}I????"?
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolutionConv2D(?A`????!???ُ??"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter!?C?????!?0?l??0"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInputA??B????!?u9????0"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_6/convolution_grad/Conv2DBackpropInputConv2DBackpropInput??CO?x??!j?-????0"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/SparseDotIncBuilder/SparseTensorDenseMatMul_grad/SparseTensorDenseMatMulSparseTensorDenseMatMulrn?e??!ѓ$????"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput?7N????!Qw?)?-??0"?
jkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder/convolutionConv2D??-??4??!?O?51??Q      Y@Y?8??8?)@a?8??8?U@q?]?];1@y??CO??"?
both?Your program is POTENTIALLY input-bound because 77.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?18.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?17.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 