?	?????;7@?????;7@!?????;7@	Ih@9;@Ih@9;@!Ih@9;@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?????;7@2s??cͼ?1ܼqR?_(@I?:?z?#@Y#ظ?]???r0*	?O??nd@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Y?X"??!?R????B@)'L5????1[+	o?.@@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???:???!ۜ?^?>@)m???L??1?	?E6@:Preprocessing2U
Iterator::Model::ParallelMapV2?'?_??!L???0@)?'?_??1L???0@:Preprocessing2F
Iterator::ModeljkD0.??!?B%??9@)?s????1i~AM	>"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??(&o???!?l*6?
@)??(&o???1?l*6?
@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?z?I|??!`?????R@)??p???1???",&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorC???-??!:Q6? @)C???-??1:Q6? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?42.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Jh@9;@I???^E@Q?Kb??9J@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	2s??cͼ?2s??cͼ?!2s??cͼ?      ??!       "	ܼqR?_(@ܼqR?_(@!ܼqR?_(@*      ??!       2      ??!       :	?:?z?#@?:?z?#@!?:?z?#@B      ??!       J	#ظ?]???#ظ?]???!#ظ?]???R      ??!       Z	#ظ?]???#ظ?]???!#ظ?]???b      ??!       JGPUYJh@9;@b q???^E@y?Kb??9J@?".
IteratorGetNext/_29_Send?<u8????!?<u8????"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter?J,?H??!?E?NO??0".
IteratorGetNext/_31_Sendg}a	Mٱ?!T??i???"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput??Q	2/??!?Ϫ????0"?
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolutionConv2D%}?ff ??!?Тwҧ??"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput??n??֤?!r???B??0"?
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolutionConv2D?j?۟??!?o	4??"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput?(??5???!??wp????0"?
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_3/convolutionConv2Df? M????!???*4???"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter?͕ڒ???!?Pv? ??0Q      Y@Y????.?(@ai??&??U@q???|L@yj?V>|??"?

device?Your program is NOT input-bound because only 4.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?42.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 