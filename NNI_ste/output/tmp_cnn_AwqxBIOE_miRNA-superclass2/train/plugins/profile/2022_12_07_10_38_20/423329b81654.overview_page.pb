?	??w﨑)@??w﨑)@!??w﨑)@	????????????!??????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??w﨑)@稣?jd??11%??%@A?^?"??}?Io???I???Y@k~??E??rEagerKernelExecute 0*	?v???d@2F
Iterator::Modelqvk?ǳ?!'??8?AG@)1?Z{????1?O??b[?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatN+?@.q??!?<???	8@)?GQg?!??1pB#d%4@:Preprocessing2U
Iterator::Model::ParallelMapV2`?n?ƙ?!J??O.@)`?n?ƙ?1J??O.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateԞ?sb??!wȿ??1@)dt@????1?I?Be?(@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??*????!lZ?b??8@)????Ӊ?1?GR??^@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?T???B??!B?4=@)?T???B??1B?4=@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipd???H???!?S?c?J@)???~?:??13???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????yz?!?h??4"@)????yz?1?h??4"@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?13.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??????I,?a??/@Q.FCʝT@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	稣?jd??稣?jd??!稣?jd??      ??!       "	1%??%@1%??%@!1%??%@*      ??!       2	?^?"??}??^?"??}?!?^?"??}?:	o???I???o???I???!o???I???B      ??!       J	@k~??E??@k~??E??!@k~??E??R      ??!       Z	@k~??E??@k~??E??!@k~??E??b      ??!       JGPUY??????b q,?a??/@y.FCʝT@?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad ;?3z??! ;?3z??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?Z?O???!`h??Ϳ?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterd? 1'ҫ?!I?:???0"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose???6?ʩ?!??M??"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	TransposeX??5c???!D??*????"3
model/Conv1D_1/BiasAddBiasAdd??W????!>?t????"1
model/Conv1D_2/conv1dConv2D??1ʒ???!.ߺN3%??"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose,??ϐt??!41?h?3??"}
^gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizer	Transpose?43???!??ϙ4??"-
model/Conv1D_1/ReluReluF8d??ͦ?!?v,???Q      Y@Y!Y?B*@a????7?U@qKR02@y??0&?Q??"?
device?Your program is NOT input-bound because only 1.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?13.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?18.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 