?	c?: ?f @c?: ?f @!c?: ?f @	??킗?@??킗?@!??킗?@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLc?: ?f @?M4?s??1t}?@A?ݰmQf??I?l??}??Y-{؜???rEagerKernelExecute 0*	
ףp=?c@2F
Iterator::Model?r.?Ue??!?NQ??H@)}?E֪?1????@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateh?4?;??!??h??{A@)R??/Ie??1k????X@@:Preprocessing2U
Iterator::Model::ParallelMapV2nē?????!i??-@)nē?????1i??-@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?g???c??!:??s??&@)Yk(?ц?1?
E"vB@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????$?{?!??AŦJ@)????$?{?1??AŦJ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??N]???!S??Z4?I@)G?˵hz?1????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapC??À??! ??8EB@)?M???Pd?1X?k?)??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?????_?!JB?ד???)?????_?1JB?ד???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???=?Z?!?R=????)???=?Z?1?R=????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 14.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?24.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??킗?@I???=%C@Q"u9y	?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?M4?s???M4?s??!?M4?s??      ??!       "	t}?@t}?@!t}?@*      ??!       2	?ݰmQf???ݰmQf??!?ݰmQf??:	?l??}???l??}??!?l??}??B      ??!       J	-{؜???-{؜???!-{؜???R      ??!       Z	-{؜???-{؜???!-{؜???b      ??!       JGPUY??킗?@b q???=%C@y"u9y	?M@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterX8?????!X8?????0"1
model/Conv1D_2/conv1dConv2D5??-ô?!?^vgi<??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputNz?x?n??!?????9??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterP!?h???! r??????0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput
??G??!cs??Ͼ??0"1
model/Conv1D_3/conv1dConv2D?&[Ȥ?!fE?R?W??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradw?oˁ&??!???oCJ??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad??K?ĝ?!?{c$?&??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??qT(??!w\}k???0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?????(??!?m??ѝ??Q      Y@YsO#,?4*@a?{a?U@q?g?j?}B@yT??????"?
both?Your program is POTENTIALLY input-bound because 14.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?24.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?37.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 