?	 |(т@ |(т@! |(т@	m?w9@m?w9@!m?w9@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL |(т@?ٕ?Q??1????@A?S?????I????9@Y{??B???rEagerKernelExecute 0*	/?$?ue@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatv4?????!?}ߦ??@@)?????1N??g?=@:Preprocessing2F
Iterator::Model?K?1?=??!y?????D@)???nI??1e??T^;@:Preprocessing2U
Iterator::Model::ParallelMapV2?x[??٘?!????E,@)?x[??٘?1????E,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??v?>X??!??M??k)@)??v?>X??1??M??k)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipD? ???!?lMe?M@)?T?e??1??'lR@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?)??Y???!zO??˛1@)??2?68??1?p??A?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor~b??U}?!]????@)~b??U}?1]????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapr?&"???!???33@)???uR_f?1?@:?t??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?30.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9m?w9@I?????NG@Q???.?}I@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ٕ?Q???ٕ?Q??!?ٕ?Q??      ??!       "	????@????@!????@*      ??!       2	?S??????S?????!?S?????:	????9@????9@!????9@B      ??!       J	{??B???{??B???!{??B???R      ??!       Z	{??B???{??B???!{??B???b      ??!       JGPUYm?w9@b q?????NG@y???.?}I@?"1
model/Conv1D_2/conv1dConv2D?i?ȷ?!?i?ȷ?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??i?H,??!dyuy???0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?%?????!N???9???0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputGh???!03?????0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputWZ???v??!{??Ҟ??0"1
model/Conv1D_3/conv1dConv2DM???d???!?vWx_v??"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?wF?????!?????$??"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilterP??b???!?
??????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad????H??!v???\u??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad?~N??r??!cl?k????Q      Y@Y?
*T?(@a?^?z??U@qrك?p3?@y??s?7??"?
both?Your program is POTENTIALLY input-bound because 15.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?30.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?31.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 