?	??r???@??r???@!??r???@	?`܁?#@?`܁?#@!?`܁?#@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??r???@???????1?t?? @A??k~????I9
3f @Yu?yƾd??rEagerKernelExecute 0*	h??|?g?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat3Q??????!V??dnNS@)?+??????1???R@:Preprocessing2F
Iterator::Model????˻?!J????/@)???_w???1?(???&@:Preprocessing2U
Iterator::Model::ParallelMapV2:w?^?"??!b???g@):w?^?"??1b???g@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?6qr???!?Ղ??@)?6qr???1?Ղ??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip^??jGq??!ןn?h	U@)@h=|?(??1̞ۡ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?T??????!
Oc
?U??)?T??????1
Oc
?U??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?P?l??!??G?@))v4????15?,???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/?KR?b??!!W;?eT@)????b)b?1?ZDg???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?26.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t14.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?`܁?#@I?@?ĪD@Q????eH@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????????????!???????      ??!       "	?t?? @?t?? @!?t?? @*      ??!       2	??k~??????k~????!??k~????:	9
3f @9
3f @!9
3f @B      ??!       J	u?yƾd??u?yƾd??!u?yƾd??R      ??!       Z	u?yƾd??u?yƾd??!u?yƾd??b      ??!       JGPUY?`܁?#@b q?@?ĪD@y????eH@?"1
model/Conv1D_2/conv1dConv2DX???y??!X???y??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?:Ze"??!??~????0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?n?0???!?R6?7???0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputS?zf???!????h???0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput򱥗???!????:???0"1
model/Conv1D_3/conv1dConv2DI6?`???!? ??n??"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?|/??!2?Ǚ?!??"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilteru}UW?՚?!	3=o???0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad??&o?Q??!B?/6$t??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad?_Jh??!J?պ???Q      Y@Y?
*T?(@a?^?z??U@q?'?b?/@y.?dd???"?
both?Your program is MODERATELY input-bound because 9.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?26.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"t14.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?15.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 