?	???8?1@???8?1@!???8?1@	???ڮ ?????ڮ ??!???ڮ ??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???8?1@??*4???1???y-@A?'eRCK?I?A?"L??Y)????B??rEagerKernelExecute 0*	??C?l?c@2F
Iterator::Modelfj?!???!???%ڇG@)3???/??1??
???>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX???ޥ?!W\?_??;@)?ٕ????1&?7??7@:Preprocessing2U
Iterator::Model::ParallelMapV21?*?ԙ?!\?v??a0@)1?*?ԙ?1\?v??a0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???;޴?!J??%xJ@)?.ޏ?/??1??Lh?#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceЙ???G??!Ĺ<??B@)Й???G??1Ĺ<??B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate0?[w???!O=???*@)\?M4???1????|?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?d??7iz?!??n? ?@)?d??7iz?1??n? ?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-AF@?#??!???\,?.@)??(&o?i?1"9??^, @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???ڮ ??I??p!?L(@Q[g@??U@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??*4?????*4???!??*4???      ??!       "	???y-@???y-@!???y-@*      ??!       2	?'eRCK??'eRCK?!?'eRCK?:	?A?"L???A?"L??!?A?"L??B      ??!       J	)????B??)????B??!)????B??R      ??!       Z	)????B??)????B??!)????B??b      ??!       JGPUY???ڮ ??b q??p!?L(@y[g@??U@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter,d$t???!,d$t???0"1
model/Conv1D_2/conv1dConv2D?d??̵?!???7c??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputI?άF???!T???H???0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad}?# ?S??!?>????"1
model/Conv1D_3/conv1dConv2D?qL????!?_?x^??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter? ?C????!??????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradw+??㖡?!b?ۋ????"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?9_o ??!???w????"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose?0?f???!?Ox؞???"3
model/Conv1D_1/BiasAddBiasAddǼ? $???!lˇ????Q      Y@YD+l$Z)@a?z2~??U@qFe???C@y???Ņϗ?"?
device?Your program is NOT input-bound because only 1.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?39.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 