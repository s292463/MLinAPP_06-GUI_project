?	?P????G@?P????G@!?P????G@	??????1@??????1@!??????1@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?P????G@????B4@1??k&?'@I?-?\74@Y?? w? @r0*	.?????@2U
Iterator::Model::ParallelMapV2\W?o?@!???i?X@)\W?o?@1???i?X@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?O7P????!>?|z????)?]P?2??1??4???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????ޣ?!??R?o???),??????1?H????:Preprocessing2F
Iterator::Model?$z?R@!?? ?y?X@)E-ͭV??1h?? $l??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?v/?ɑ?!?h??>??)?v/?ɑ?1?h??>??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?'?8??!???w?a??)DOʤ???1??0w?K??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorKi????!??????)Ki????1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 18.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?42.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t14.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??????1@I"r??7?L@Q-n@?e8@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????B4@????B4@!????B4@      ??!       "	??k&?'@??k&?'@!??k&?'@*      ??!       2      ??!       :	?-?\74@?-?\74@!?-?\74@B      ??!       J	?? w? @?? w? @!?? w? @R      ??!       Z	?? w? @?? w? @!?? w? @b      ??!       JGPUY??????1@b q"r??7?L@y-n@?e8@?".
IteratorGetNext/_25_Send`?=t???!`?=t???".
IteratorGetNext/_29_Send??0'??!1^.????".
IteratorGetNext/_27_SendS??/b/??!?Uz?޶??".
IteratorGetNext/_31_Send&????A??!5?????"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter] "a좒?!U ?	??0"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInputq??Ґ?!)?me_??0"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_6/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter??????!??[d???0"?
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_4/convolutionConv2D?՜&z??!B)?i&??"?
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_6/convolutionConv2D?!?????!A??~??"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_5/convolution_grad/Conv2DBackpropInputConv2DBackpropInputr??ԯ??!??X?????0Q      Y@YVUUUU?'@aVUUUUV@q"!;???@y???& )??"?
both?Your program is MODERATELY input-bound because 18.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?42.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"t14.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 