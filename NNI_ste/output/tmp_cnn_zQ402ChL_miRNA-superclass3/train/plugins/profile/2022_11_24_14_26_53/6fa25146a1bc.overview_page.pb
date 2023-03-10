?	|G?	1?@|G?	1?@!|G?	1?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC|G?	1?@fI??Z6??1?d??~#??A{fI??Z??I%??@rEagerKernelExecute 0*	gffff?c@2F
Iterator::Model? ????!???y?	I@)H2?w???1f????S@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??	K<???!??O<h?9@)L???H???1;??Df?5@:Preprocessing2U
Iterator::Model::ParallelMapV2??iܛ?!}
`?/k1@)??iܛ?1}
`?/k1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?v5yʚ?!????0@)+?&?|???1?8i?%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?v?k??!J?/??@)?v?k??1J?/??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?{?i????!\Md?~?H@)M??u??1<?ƪ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorbg
??x?!1һ@)bg
??x?11һ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?uq???!&??叭2@)ޏ?/??h?1!GN???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 25.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?49.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI5??b¶R@Q+?ht?$9@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	fI??Z6??fI??Z6??!fI??Z6??      ??!       "	?d??~#???d??~#??!?d??~#??*      ??!       2	{fI??Z??{fI??Z??!{fI??Z??:	%??@%??@!%??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q5??b¶R@y+?ht?$9@?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputl??w ڼ?!l??w ڼ?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??	??I??!+?_???0"1
model/Conv1D_2/conv1dConv2D-??????!;ca????"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?k?Dm???!???	Ӊ??0"1
model/Conv1D_3/conv1dConv2D~???C ??!?9\{?	??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?> ^????!?A ??i??0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradǻ%??I??!6?D??r??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradl?u2t??!?ߡj??"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad֊|?5??!???c=??"1
model/Conv1D_1/conv1dConv2D????J'??!?9?w??Q      Y@Y&W?+?)@a?????U@q??:2@S@y??u????"?
both?Your program is POTENTIALLY input-bound because 25.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?49.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?77.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 