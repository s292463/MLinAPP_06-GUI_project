?	??%??1@??%??1@!??%??1@	ZD??6u??ZD??6u??!ZD??6u??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??%??1@?v?>X???1?>?'I'-@I{?<d??@Y???vhX??rEagerKernelExecute 0*	?l???A~@2F
Iterator::Model?]?)ʥ??!?J?S@)Wzm6Vb??1!QwTBAQ@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǞ=??I??!?"?:?^0@)?52;???1H?_?-@:Preprocessing2U
Iterator::Model::ParallelMapV2??????!??LZ?8@)??????1??LZ?8@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicewLݕ]0??!f?b???@)wLݕ]0??1f?b???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??? ?!?y~8?+@)????S??1?i????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipIh˹W??!X??׶?7@)??̔?߂?1?"il?u??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorҏ?S??{?!ﯭ????)ҏ?S??{?1ﯭ????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaph????Ś?!?PC?P?@){??h?1ع&??u??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?13.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ZD??6u??I??VU4/@Q?Ū???T@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?v?>X????v?>X???!?v?>X???      ??!       "	?>?'I'-@?>?'I'-@!?>?'I'-@*      ??!       2      ??!       :	{?<d??@{?<d??@!{?<d??@B      ??!       J	???vhX?????vhX??!???vhX??R      ??!       Z	???vhX?????vhX??!???vhX??b      ??!       JGPUYZD??6u??b q??VU4/@y?Ū???T@?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilteri?o?m??!i?o?m??0"1
model/Conv1D_2/conv1dConv2D4R???ƺ?!??H?@??"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterۋ&/{??!??????0"1
model/Conv1D_3/conv1dConv2D??	?e???!?K?"??"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputp?۩{(??!5?JH???0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput??]???!?"tS???0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad?"5Q??!6LU?c??"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad??	?e???!?? z:??"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?B9????!??;?Z\??0"3
model/Conv1D_1/BiasAddBiasAdd??&?9n??!?Τ????Q      Y@YAd?W?,)@ax??g?U@q???! @y?"%?:F??"?

device?Your program is NOT input-bound because only 1.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?13.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 