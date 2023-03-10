?	Kr??&/(@Kr??&/(@!Kr??&/(@	??u41@??u41@!??u41@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLKr??&/(@6 B\9;@1C9Ѯb@A333333??I?׃I??@YM???? @rEagerKernelExecute 0*	????/f@2F
Iterator::Model?7?0???!??O?F@)??up?7??1??????;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ӹ????!u?h';@)*???O??1?~ST?97@:Preprocessing2U
Iterator::Model::ParallelMapV2^????4??!3J???1@)^????4??13J???1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?_=?[???!?N?c??%@)?_=?[???1?N?c??%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipe?fb???!????5K@)U??-?|??1́oZ0>#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateV?@?)V??!???u?#0@)?|ԛQ??1?9??A@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?\????{?!մ???@)?\????{?1մ???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapG?0}?!??!?I?>"?1@)???ig?1(;?,???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 17.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?22.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t39.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??u41@Iga?m?`O@QJ8?_
4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	6 B\9;@6 B\9;@!6 B\9;@      ??!       "	C9Ѯb@C9Ѯb@!C9Ѯb@*      ??!       2	333333??333333??!333333??:	?׃I??@?׃I??@!?׃I??@B      ??!       J	M???? @M???? @!M???? @R      ??!       Z	M???? @M???? @!M???? @b      ??!       JGPUY??u41@b qga?m?`O@yJ8?_
4@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter8??~???!8??~???0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputE]??#???!??X?PA??0"1
model/Conv1D_2/conv1dConv2DTM??<???!??妷???"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??4w????!?2??\??0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput???c??!??L?????0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrady?/???!?0?L???"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad?????[??!.}	Ha??"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrads?Ҳ???!ե6?D!??"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits?V?????!A??f???"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???ȗ???!L?V??0Q      Y@YH?R&?&@a??5?'V@q?V???@y?j?g???"?
both?Your program is MODERATELY input-bound because 17.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?22.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t39.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 