?	?Yh?4s @?Yh?4s @!?Yh?4s @      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?Yh?4s @?J??????1v?և?f@A?_??D??I?fh<@rEagerKernelExecute 0*	<?O???u@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateA?C???!Y5$??Q@)???}??1???u?CQ@:Preprocessing2F
Iterator::ModeltϺFˁ??!,O???4@)H??0~??19?+_**@:Preprocessing2U
Iterator::Model::ParallelMapV2@/ܹ0қ?!????!?@)@/ܹ0қ?1????!?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?_???ܓ?!݉?.m~@)?7?{?5??1??? ?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?*Q??r??!5?O???S@)Y?+???~?1r??#?T@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??#?{?!<?l@s???)??#?{?1<?l@s???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6????t??!c?B??Q@)Z???аh?1???????:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorR?=?Ne?!&K(??!??)R?=?Ne?1&K(??!??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????_?!??????)?????_?1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 24.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?48.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???ccR@Q???qr:@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?J???????J??????!?J??????      ??!       "	v?և?f@v?և?f@!v?և?f@*      ??!       2	?_??D???_??D??!?_??D??:	?fh<@?fh<@!?fh<@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???ccR@y???qr:@?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput???????!???????0"1
model/Conv1D_2/conv1dConv2D??????!$t6H???"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter%l?c??!?U??Q??0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???????!??r/B??0"1
model/Conv1D_3/conv1dConv2D?Y??e??!?Q??????"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad????iT??!X0e05t??"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput???9??!?0$??7??0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?>x????!????????"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad??z????!?_?NJ??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradذ??CG??!Kcþ??Q      Y@Yp???*@a??Ǐ?U@qlUE{?ZC@y????????"?
both?Your program is POTENTIALLY input-bound because 24.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?48.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?38.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 