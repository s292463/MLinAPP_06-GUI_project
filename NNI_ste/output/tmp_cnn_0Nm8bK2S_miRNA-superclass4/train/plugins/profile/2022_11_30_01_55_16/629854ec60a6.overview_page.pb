?	5???2=@5???2=@!5???2=@	??*VZ????*VZ??!??*VZ??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC5???2=@	???k??1V)=?K4/@I?R????)@Y?g??`o??rEagerKernelExecute 0*	??v???c@2F
Iterator::Model?RAEկ??!???D?I@)??"???1W?҆1@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatϡU1???!If??<@)?r0? â?1???vJ7@:Preprocessing2U
Iterator::Model::ParallelMapV2????ȑ??!???{z?2@)????ȑ??1???{z?2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?	1?Tm??!???n@)?	1?Tm??1???n@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateh?????!???|N+@)C?l搄?1x?8?݇@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip=?[????!B??QH@)?L????1K?U
?0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorz?΅?~?!??U?P?@)z?΅?~?1??U?P?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaph??????![Hׯ??.@)R?=?Ne?1<??mys??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?43.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??*VZ??I?????F@QB??]??J@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
		???k??	???k??!	???k??      ??!       "	V)=?K4/@V)=?K4/@!V)=?K4/@*      ??!       2      ??!       :	?R????)@?R????)@!?R????)@B      ??!       J	?g??`o???g??`o??!?g??`o??R      ??!       Z	?g??`o???g??`o??!?g??`o??b      ??!       JGPUY??*VZ??b q?????F@yB??]??J@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?2?|????!?2?|????0"1
model/Conv1D_2/conv1dConv2D4C?v???!???0??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?&???!?@z?1??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilteraO????!.?\?y???0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradp?[IK??!?jϷ???"1
model/Conv1D_3/conv1dConv2D?JbC???!?s$???"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?]???!m??????"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose'書?*??!|+~#????"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose7??9???!Wqh??"3
model/Conv1D_1/BiasAddBiasAddJ	f??T??!?y?=??Q      Y@Y@n]?G*@a8R4??U@q??L??!@y?O6????"?

device?Your program is NOT input-bound because only 1.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?43.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 