?	(??Z&?@(??Z&?@!(??Z&?@	?򗧀@?򗧀@!?򗧀@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL(??Z&?@???p????1??&????A??mRј?Is??/ٸ@Y?ާ??@??rEagerKernelExecute 0*	o??ʍu@2F
Iterator::ModelU?	g????!??K?@?R@)?e?%????1ҽ?-ZO@:Preprocessing2U
Iterator::Model::ParallelMapV2u<f?2???!??jR?(@)u<f?2???1??jR?(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??	L?u??!???Y?
&@)DOʤ?6??1??iL]"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate62;?ޡ?!???V?=$@)?XP?i??1)M?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicec'????!C??#(\@)c'????1C??#(\@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?,z????!R?Њ??8@)??]?9???1??^???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorQ??9?y?!?7??Wk??)Q??9?y?1?7??Wk??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap]??k??!~?K?H?%@)w?E?h?1m?u7??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?50.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t22.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?򗧀@I?[G?xQR@Qz????Y4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???p???????p????!???p????      ??!       "	??&??????&????!??&????*      ??!       2	??mRј???mRј?!??mRј?:	s??/ٸ@s??/ٸ@!s??/ٸ@B      ??!       J	?ާ??@???ާ??@??!?ާ??@??R      ??!       Z	?ާ??@???ާ??@??!?ާ??@??b      ??!       JGPUY?򗧀@b q?[G?xQR@yz????Y4@?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?	Og<??!?	Og<??0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterd???Ƶ?!?cn+???0"1
model/Conv1D_2/conv1dConv2D???=?n??!L?i????"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFiltergVa??=??!6?X???0"1
model/Conv1D_3/conv1dConv2D????9???!W3????"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput???H?9??!??Q??2??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad[?i~$??!:h67??"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?N??!?v; 9??"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad`vfY??!?u??????"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradx#?q8??!?'????Q      Y@Y&W?+?)@a?????U@q??l?2mA@y?1?~???"?
both?Your program is MODERATELY input-bound because 6.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?50.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t22.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?34.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 