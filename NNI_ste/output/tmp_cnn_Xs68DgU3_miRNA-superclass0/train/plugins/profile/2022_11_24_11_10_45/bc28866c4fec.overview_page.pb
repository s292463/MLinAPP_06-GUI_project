?	?`?H?&@?`?H?&@!?`?H?&@	A???@A???@!A???@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?`?H?&@?|гY???1?W?\?@AԷ?鲘??Ip?71$?@Y?,??o???rEagerKernelExecute 0*	?ʡE?3e@2F
Iterator::Model?<֌??!?&5O?cI@)Wzm6Vb??1?Z?^~A@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?wg????!???????@)??2nj???1-?bN?=@:Preprocessing2U
Iterator::Model::ParallelMapV2????m??!?.?dF?/@)????m??1?.?dF?/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??{?专?!BE??ʵ&@)7U??檉?1?Tlja?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?"?ng_??!J?ʰO?H@)????L0|?1??@!?:@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorT? Pō{?!?k?Ng?@)T? Pō{?1?k?Ng?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap#?-?R\??!±g*??@@)J?i?WVj?1????S??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorH??'??c?!??????)H??'??c?1??????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?0{?v?Z?!Gq??????)?0{?v?Z?1Gq??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?20.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t12.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9A???@I??'J\?@@Qk??"ЕN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?|гY????|гY???!?|гY???      ??!       "	?W?\?@?W?\?@!?W?\?@*      ??!       2	Է?鲘??Է?鲘??!Է?鲘??:	p?71$?@p?71$?@!p?71$?@B      ??!       J	?,??o????,??o???!?,??o???R      ??!       Z	?,??o????,??o???!?,??o???b      ??!       JGPUYA???@b q??'J\?@@yk??"ЕN@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter{/?!a???!{/?!a???0"1
model/Conv1D_2/conv1dConv2D??`?X??!&:ƨ???"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?s?#f???!??10???0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?`괷͠?!%#^(???"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?ߨ;???!?"z?+??0"1
model/Conv1D_3/conv1dConv2D?"[??!?/?O????"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?s?F???!Xj?????"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad?e9????!YH?p?(??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?5?????!???w????0"C
%gradient_tape/model/Conv1D_2/ReluGradReluGradٛ?!b???!???????Q      Y@Yݘ??V+@a`?.?U@qV?$??7@y?/1M??"?
both?Your program is MODERATELY input-bound because 5.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?20.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"t12.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?23.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 