?	?1!撺5@?1!撺5@!?1!撺5@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?1!撺5@T?^PZ?1?»\?7@I?T13@r0*	t?V~q@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?G?)s???!??qI@)?G?)s???1??qI@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatAJ???n??!?2?tO?1@)R???????1??g?(-@:Preprocessing2U
Iterator::Model::ParallelMapV2(?8'0??!n? ?^$@)(?8'0??1n? ?^$@:Preprocessing2F
Iterator::Model?$#gaO??!?%???3@)i???n??1??c? ?!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?5[y????!?v?X<T@)??հߓ?1,?X?k?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?2??(??!?5?{|!L@)X??C???1v???T?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?ߡ(?'??!5 j?V	@)?ߡ(?'??15 j?V	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?87.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI}z1???U@Q,t.?i(@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	T?^PZ?T?^PZ?!T?^PZ?      ??!       "	?»\?7@?»\?7@!?»\?7@*      ??!       2      ??!       :	?T13@?T13@!?T13@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q}z1???U@y,t.?i(@?".
IteratorGetNext/_30_Recv?h??Ny??!?h??Ny??"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput?S?5????!??H?w??0"?
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolutionConv2D:t?$A~??!??@???"?
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolutionConv2D??+j-??!??<\?o??"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter?Ȑ wʔ?!?N<<	??0"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput5??+??!H??}]???0"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/SparseDotIncBuilder/SparseTensorDenseMatMul_grad/SparseTensorDenseMatMulSparseTensorDenseMatMul??$f????!=???"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_6/convolution_grad/Conv2DBackpropInputConv2DBackpropInputG1ώYP??!%*???0"?
?gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_581/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInputĂ?#?đ?!Q?go7??0"?
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_5/convolutionConv2D?/?@??!??n??'??Q      Y@Y?8??8?)@a?8??8?U@q}G?'?(*@y߹??????"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?87.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?13.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 