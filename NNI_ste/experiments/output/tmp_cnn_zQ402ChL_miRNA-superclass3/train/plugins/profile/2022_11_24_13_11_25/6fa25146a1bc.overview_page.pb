?	Zd;?O?$@Zd;?O?$@!Zd;?O?$@	+?^??
@+?^??
@!+?^??
@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLZd;?O?$@G?,?@1????g2@A?ۄ{eު?I>{.S??@Y?N\?W ??rEagerKernelExecute 0*	 ??Q??u@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateQ??lu??!R??FWnP@)f`X???1??B?+P@:Preprocessing2F
Iterator::Model??????!?K??~8@)???-??1&??*@:Preprocessing2U
Iterator::Model::ParallelMapV2P÷?n???!???K#&@)P÷?n???1???K#&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?;?????!
mUx?R@)?,{،?16.W?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?s????!/??8?@)ۤ?????1?n?
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŭ???w?!?,?bo???)ŭ???w?1?,?bo???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???9???!?@V???P@)?P?[?e?1ә??)5??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor??????]?!Ԡ	????)??????]?1Ԡ	????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?J?E?]?!RyEYe???)?J?E?]?1RyEYe???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 28.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?47.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9*?^??
@I??B?'S@Q??(???4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	G?,?@G?,?@!G?,?@      ??!       "	????g2@????g2@!????g2@*      ??!       2	?ۄ{eު??ۄ{eު?!?ۄ{eު?:	>{.S??@>{.S??@!>{.S??@B      ??!       J	?N\?W ???N\?W ??!?N\?W ??R      ??!       Z	?N\?W ???N\?W ??!?N\?W ??b      ??!       JGPUY*?^??
@b q??B?'S@y??(???4@?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput??g????!??g????0"1
model/Conv1D_2/conv1dConv2Dڄq????!???ڭ??"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter~???
_??!&Bj-?n??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter\aW????!R.Z??b??0"1
model/Conv1D_3/conv1dConv2D??D???!҇?Q???"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputӸLl̯??!_S????0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?#??Ԅ??!??jhaz??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad	?e?蕘?!??????"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad4???!P?G?}??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad?cW??v??!?H0???Q      Y@Yp???*@a??Ǐ?U@q? ɂN=@y?"?Y???"?
both?Your program is POTENTIALLY input-bound because 28.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?47.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?29.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 