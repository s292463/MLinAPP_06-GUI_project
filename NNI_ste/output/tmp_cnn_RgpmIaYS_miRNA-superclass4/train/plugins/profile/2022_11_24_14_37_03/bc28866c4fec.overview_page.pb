?	?bd?/4@?bd?/4@!?bd?/4@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?bd?/4@HP????1??A%??/@A?}(F???I^?SH?@rEagerKernelExecute 0*	?G?z f@2F
Iterator::Model<i??
??!?Z??mlI@)?ٕ????1 ?ʥP=@:Preprocessing2U
Iterator::Model::ParallelMapV2O]?,σ??!$???5?5@)O]?,σ??1$???5?5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatQ?l???!?(k708@)&?lscz??1????Zc4@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceT?n.????!wϧk?**@)T?n.????1wϧk?**@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}(F??!l?	??H@):ZՒ???1??7?y@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatew?ِf??!???{2@)5??,??1???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorT? Pō{?!?"4??f@)T? Pō{?1?"4??f@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??72????!<5??3@)?3??`i?1?aԯ? ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???V<5@QZxY??S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	HP????HP????!HP????      ??!       "	??A%??/@??A%??/@!??A%??/@*      ??!       2	?}(F????}(F???!?}(F???:	^?SH?@^?SH?@!^?SH?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???V<5@yZxY??S@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?:??E*??!?:??E*??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??v?м??!݂ ?s??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???-????!R?~[D???0"1
model/Conv1D_2/conv1dConv2D?Ot?*???!D???i_??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput?K??o??!???iM??0"1
model/Conv1D_3/conv1dConv2D??5?????!?D????"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?A?????!JVԎ???0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad>u??ԟ?!?V?7h>??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradx??ӝ?!?{WX-??"3
model/Conv1D_1/BiasAddBiasAdd5[K???![U?j????Q      Y@Y?]??O?'@aX,
V@qϘ??t24@y (??kj??"?
both?Your program is POTENTIALLY input-bound because 9.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?20.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 