?	?7?ܘ?@?7?ܘ?@!?7?ܘ?@	?a??:
@?a??:
@!?a??:
@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?7?ܘ?@?&P?"???1?????@AVJ??ci?I?Z?}??Y???y0??rEagerKernelExecute 0*	D?l??Qc@2F
Iterator::Model}??A?<??!???a??I@)܁:?ѭ?1??J?B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?֍wG??!?(Q?1'<@)膦?????1Ɍ|A?e7@:Preprocessing2U
Iterator::Model::ParallelMapV2ٕ??zO??!?????*@)ٕ??zO??1?????*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?V?????!?H??R?@)?V?????1?H??R?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???͎T??!;>^?@mH@)?$??ڄ?1?
??iZ@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?N?z1???!?9<[??(@)?G?RE??1?*?)?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??|	~?!oR+*@)??|	~?1oR+*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapQ3???U??!?"??i9,@)?"?~?f?1FE/?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?21.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?a??:
@I???/?q:@Q?)ǑQ@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?&P?"????&P?"???!?&P?"???      ??!       "	?????@?????@!?????@*      ??!       2	VJ??ci?VJ??ci?!VJ??ci?:	?Z?}???Z?}??!?Z?}??B      ??!       J	???y0?????y0??!???y0??R      ??!       Z	???y0?????y0??!???y0??b      ??!       JGPUY?a??:
@b q???/?q:@y?)ǑQ@?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???mJh??!???mJh??0"1
model/Conv1D_3/conv1dConv2Di???$Ψ?!???7???"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter&?4??M??!ؚ{a??0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput?Y94Al??!L?&?!???0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??n
???!4??O????0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?.;???!?3 ??0"1
model/Conv1D_2/conv1dConv2D?B??
???!???8zn??"1
model/Conv1D_4/conv1dConv2D???5???!????7??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradS?N`?ݙ?!?r??????"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad???
K??!???4?I??Q      Y@YAd?W?,)@ax??g?U@q?\?B?D8@y?۰$L??"?
both?Your program is POTENTIALLY input-bound because 4.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?21.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?24.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 