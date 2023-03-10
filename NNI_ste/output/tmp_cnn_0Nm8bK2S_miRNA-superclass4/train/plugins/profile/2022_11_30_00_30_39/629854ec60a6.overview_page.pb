?	??_#?%@??_#?%@!??_#?%@	?kP\w&@?kP\w&@!?kP\w&@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??_#?%@5S?@1m?kA@A???)u?I?d???@Y????B??rEagerKernelExecute 0*	+???5?@2U
Iterator::Model::ParallelMapV2q;4,F]??!??KD8?H@)q;4,F]??1??KD8?H@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate8??9??!?j?f?m?@)IZ??c??1?P@cv<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?J̳?V??!~n?&?#@)M??u???1+WjjX?!@:Preprocessing2F
Iterator::Model4???????! L@)>"?D??1?c?>'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice@??>??!а5??@)@??>??1а5??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?ʼUס??!??????E@)?0????1??ss?[ @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?'??Q|?!/?:`?l??)?'??Q|?1/?:`?l??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??f??}??!?hB?;??@)?^ q?1M??Q/w??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?21.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t21.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?kP\w&@IJ6XvtE@Q????G@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	5S?@5S?@!5S?@      ??!       "	m?kA@m?kA@!m?kA@*      ??!       2	???)u????)u?!???)u?:	?d???@?d???@!?d???@B      ??!       J	????B??????B??!????B??R      ??!       Z	????B??????B??!????B??b      ??!       JGPUY?kP\w&@b qJ6XvtE@y????G@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??]Ğ???!??]Ğ???0"1
model/Conv1D_2/conv1dConv2D]3??ȭ?!???T????"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad??_	?&??!?|S?:???"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad㽒h?k??!&????"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transposew??腥?!5?Z?ʽ??"3
model/Conv1D_1/BiasAddBiasAdd???\G??!?f??"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose??l?9??!~r?????"-
model/Conv1D_1/ReluRelu?	?{???!?3~ ͊??"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transposeא	ǒ??!?N?&???"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilterl???/??!???c??0Q      Y@YD+l$Z)@a?z2~??U@qO???y
@yZ???zӳ?"?
both?Your program is MODERATELY input-bound because 11.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?21.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t21.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 