?	?j?=&?@?j?=&?@!?j?=&?@	??/??@??/??@!??/??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?j?=&?@8h?>z??1q?-?S@A??V_]??I??4`?? @Y=|?(B???rEagerKernelExecute 0*	?~j?t7q@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate_ Q??!?ۛ}4#G@))??????1??3?_E@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???????!x6?eĦ6@)?̒ 5???1?	??Z4@:Preprocessing2F
Iterator::Model?n?EE???!?ߜC?c:@)vl?u???1?OX?"3@:Preprocessing2U
Iterator::Model::ParallelMapV2P?R)v??!L?2??@)P?R)v??1L?2??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice4??????!?`B???@)4??????1?`B???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?AҧU???!?/gR@)B?Ѫ?t??1o???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor!O!W?y?!?f??_@)!O!W?y?1?f??_@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??Cl???!? ??G@)ŭ???g?1??O????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?27.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??/??@I{2S?DF@Q?ʮ{QJ@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	8h?>z??8h?>z??!8h?>z??      ??!       "	q?-?S@q?-?S@!q?-?S@*      ??!       2	??V_]????V_]??!??V_]??:	??4`?? @??4`?? @!??4`?? @B      ??!       J	=|?(B???=|?(B???!=|?(B???R      ??!       Z	=|?(B???=|?(B???!=|?(B???b      ??!       JGPUY??/??@b q{2S?DF@y?ʮ{QJ@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?U???!?U???0"1
model/Conv1D_2/conv1dConv2D????ڳ?!f?}???"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?&??e???!	Ϡ??O??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad???????!?+????"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad?<LN@D??!???
???"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter*?????!???҇???0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposeMz?;#??!@(6???"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?!g?Ӫ??!|?ԬI&??"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose???)L??!?K???/??"3
model/Conv1D_1/BiasAddBiasAdd??MF??!??M????Q      Y@Y@n]?G*@a8R4??U@qY??z?;@y???'W??"?
both?Your program is POTENTIALLY input-bound because 16.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?27.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?27.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 