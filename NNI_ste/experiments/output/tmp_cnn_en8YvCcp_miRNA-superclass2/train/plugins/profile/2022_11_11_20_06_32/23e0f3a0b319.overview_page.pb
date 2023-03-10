�	�\��ʁ@�\��ʁ@!�\��ʁ@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�\��ʁ@~�{�Fv@1EJ�y�-b@A4Փ�G߰?I����P@rEagerKernelExecute 0*	��n�Hc@2F
Iterator::ModeltA}˜.�?!�{}�5IH@)m���L�?1�/U1��@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate*�=%�Ħ?!��w���<@)��P1�ߤ?1�u��m:@:Preprocessing2U
Iterator::Model::ParallelMapV2����5"�?!_.�|�.@)����5"�?1_.�|�.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�<e5]O�?!u��oʶI@)üǙ&l�?1����@�#@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�G�RE�?! q���%@)8�L��?12EK���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorX�|[�Tw?!�u-���@)X�|[�Tw?1�u-���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap,f��!�?!��%��>@)�k$	�e?1U_����?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensory=��`?!wT�N�?)y=��`?1wT�N�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�iT�d[?!����jW�?)�iT�d[?1����jW�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 62.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���	�R@Q��wً9@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	~�{�Fv@~�{�Fv@!~�{�Fv@      ��!       "	EJ�y�-b@EJ�y�-b@!EJ�y�-b@*      ��!       2	4Փ�G߰?4Փ�G߰?!4Փ�G߰?:	����P@����P@!����P@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���	�R@y��wً9@�"1
model/Conv1D_2/conv1dConv2D_�Y9݉�?!_�Y9݉�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�qp6P�?!����	m�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��sp5��?!BďF���?0"1
model/Conv1D_3/conv1dConv2D����ң�?!79�Y�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterKg�,.��?! ��cr��?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput�'��z(�?!+���.�?0"1
model/Conv1D_4/conv1dConv2D`W���?!P����?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��\��ׇ?!�9g�?0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput�#QJ�l�?!�?2�_�?0"1
model/Conv1D_1/conv1dConv2D��<��?!�2?6���?Q      Y@Y
��%9@aX߇�m�W@q󯆹��J@y�8�fJ?"�
both�Your program is POTENTIALLY input-bound because 62.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�53.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 