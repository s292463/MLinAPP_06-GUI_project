? 	??ל@??ל@!??ל@	ox???@ox???@!ox???@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??ל@H?c?C???1Q.?_x???A????W??ILݕ]0X@Yx$(~???rEagerKernelExecute 0*	???w??@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??????!?`???<I@)Ǆ?K????1X???#H@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map???????!`rE??F@)?#?&ݖ??1?p?5?D@:Preprocessing2F
Iterator::Model?rf?B??!9?މ]?@)3???/??1??E?3@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??+d???!?`e6.@)?I|????1iBt&? @:Preprocessing2U
Iterator::Model::ParallelMapV2?խ??ޗ?!-?1?v]??)?խ??ޗ?1-?1?v]??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate'???????!'?a????)E?A???1?JC?????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?r?4???!B??l???)??r????1`???\???:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?<Fy????!0y_?????)?<Fy????10y_?????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?x?ߢ???!????J@)?:?f??1?-wd?y??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?Hh˹w?!I??9????)?Hh˹w?1I??9????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceb?? ??t?!?<	?e8??)b?? ??t?1?<	?e8??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range#?ng_yp?!??^?????)#?ng_yp?1??^?????:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate????	??!>??\???)2?w?f?1?($?|??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorHP?s?R?!?屯??)HP?s?R?1?屯??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?59.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ox???@I????$T@Qߺ?4?t.@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	H?c?C???H?c?C???!H?c?C???      ??!       "	Q.?_x???Q.?_x???!Q.?_x???*      ??!       2	????W??????W??!????W??:	Lݕ]0X@Lݕ]0X@!Lݕ]0X@B      ??!       J	x$(~???x$(~???!x$(~???R      ??!       Z	x$(~???x$(~???!x$(~???b      ??!       JGPUYox???@b q????$T@yߺ?4?t.@?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad??z?????!??z?????"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter$? h~??!???????0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits??B??ӧ?!??@?Zo??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?M.???!?"?U&???0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???[꿜?!?U??#???0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???^H???!>EmVf???0"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad*??ȷ???!?????C??"1
model/Conv1D_2/conv1dConv2DK?J?????!6??>M???"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput>??????!ʓ???x??0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad??Vp?f??!????P??Q      Y@Y??????2@aKKKKKKT@q!J?R??+@y@1.????"?
both?Your program is POTENTIALLY input-bound because 20.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?59.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?13.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 