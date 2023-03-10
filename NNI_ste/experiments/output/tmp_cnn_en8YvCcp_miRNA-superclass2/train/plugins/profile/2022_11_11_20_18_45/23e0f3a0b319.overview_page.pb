�	��h@��h@!��h@	��-��p�?��-��p�?!��-��p�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��h@N`:���Z@1eq����M@AjN^d~�?Ifj�!�<@Y(��9x&�?rEagerKernelExecute 0*	`��"�{@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��/E�?!�Z�r�T@)�(\����?1е�xx�S@:Preprocessing2F
Iterator::Model� �X4��?!�dѠ�*@)]���lȟ?1�C�n��@:Preprocessing2U
Iterator::Model::ParallelMapV2����q�?!��^3��@)����q�?1��^3��@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�T� ��?!�`ת��@)�zi� ��?1����K@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipI�[��?!is��˪U@)-y<-?�?1pBOA�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor|,}��v?!XQy�;��?)|,}��v?1XQy�;��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��#�k?!�����T�?)��#�k?1�����T�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapW|C�u�?!H�s�8T@)D���XPh?1ץ�P ��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice8���CY?![-���?)8���CY?1[-���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 54.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�14.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��-��p�?I�ءz�dQ@Q�o��\>@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	N`:���Z@N`:���Z@!N`:���Z@      ��!       "	eq����M@eq����M@!eq����M@*      ��!       2	jN^d~�?jN^d~�?!jN^d~�?:	fj�!�<@fj�!�<@!fj�!�<@B      ��!       J	(��9x&�?(��9x&�?!(��9x&�?R      ��!       Z	(��9x&�?(��9x&�?!(��9x&�?b      ��!       JGPUY��-��p�?b q�ءz�dQ@y�o��\>@�"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputq��_ʼ�?!q��_ʼ�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��u2j�?!��j~��?0"1
model/Conv1D_3/conv1dConv2D���h��?!�0�kY��?"1
model/Conv1D_2/conv1dConv2Ds#�z8�?!���h�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��|
��?!�0�q9��?0"1
model/Conv1D_4/conv1dConv2Dì.���?!��w�r��?"1
model/Conv1D_1/conv1dConv2D��Z�{��?!(�-�J��?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��E�X#�?!5��	�?0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput���jJg�?!h�d���?0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�:w��?!y8�8���?0Q      Y@Y9#x��@a�}���W@q"m��z5@y�$����_?"�
both�Your program is POTENTIALLY input-bound because 54.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�14.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�21.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 