?	?z?p?z%@?z?p?z%@!?z?p?z%@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?z?p?z%@?`?d@1v??X??@AI?p??I?=?NU@rEagerKernelExecute 0*	1?Z?d@2F
Iterator::Model????R??!֘?k?H@)?m????1?????Q@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeats??P???!A?A~??9@)m??????1???Ϗ?5@:Preprocessing2U
Iterator::Model::ParallelMapV2l?u????!A???D1@)l?u????1A???D1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateۣ7?Gn??!???81@)#?ng_y??1ȋ???G#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceq???????!?F;?S@)q???????1?F;?S@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?N$?jf??!*g??I@)?/??????1m??U?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor0h!??{?!Һ?C@)0h!??{?1Һ?C@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??e?O7??!y,[?l?2@)?]???h?1?N?4???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 31.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?40.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI????R@Q???dx?;@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?`?d@?`?d@!?`?d@      ??!       "	v??X??@v??X??@!v??X??@*      ??!       2	I?p??I?p??!I?p??:	?=?NU@?=?NU@!?=?NU@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????R@y???dx?;@?"1
model/Conv1D_3/conv1dConv2D???fl??!???fl??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter@
?uK??!??ʺ?[??0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???yV???!+YX???0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputc???:y??!5??Z??0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput?6 ???!{Qq??!??0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad??p?S??!?^8????"1
model/Conv1D_2/conv1dConv2D?+?cK??!?+????"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad?m??@??!x?Sޜ???"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad|Iۍ?R??!?1??d??"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad???|:??!???>i??Q      Y@Y      )@a     ?U@qRɬ???J@yY>??a??"?
both?Your program is POTENTIALLY input-bound because 31.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?40.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?54.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 