�	���$�@���$�@!���$�@	�ԠE��@�ԠE��@!�ԠE��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL���$�@�2��Y�?1��<e@AiUMuO?I|`���	@Y�����y�?rEagerKernelExecute 0*	䥛� �f@2F
Iterator::Model��g?RD�?!
���H@)��3��X�?1ͦ�r_@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatr3܀��?!Os��u3=@)ƥ*mq��?14t3j9@:Preprocessing2U
Iterator::Model::ParallelMapV2E.8��_�?!<���.@)E.8��_�?1<���.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�
���?!��si�I@)����?1?4�s"@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceƦ�B ��?!�T�@)Ʀ�B ��?1�T�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�f�C�?!Cj˄�(@)�&�%��?1��BaH�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoram���|?!���K@)am���|?1���K@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����?!���a�	+@)���9]f?1�M���!�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 2.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�40.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�ԠE��@I��5f=D@Q���AiL@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�2��Y�?�2��Y�?!�2��Y�?      ��!       "	��<e@��<e@!��<e@*      ��!       2	iUMuO?iUMuO?!iUMuO?:	|`���	@|`���	@!|`���	@B      ��!       J	�����y�?�����y�?!�����y�?R      ��!       Z	�����y�?�����y�?!�����y�?b      ��!       JGPUY�ԠE��@b q��5f=D@y���AiL@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�䜊�B�?!�䜊�B�?0"1
model/Conv1D_2/conv1dConv2D�~���?!z�C`��?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput��\:]��?!j��w��?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad?	^^U�?!�M�tF��?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad���[�?! �>4�C�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter}��NС?!Pb�}�?0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter@�	�?!$tN�an�?0"3
model/Conv1D_1/BiasAddBiasAdd�%{p���?!�&Vf-Y�?"-
model/Conv1D_1/ReluRelu��_�b�?!�!��c'�?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose~*�FV�?!�D&���?Q      Y@Y��u@7�)@a%D�9�U@qLi�;H;@y 1?�4#�?"�
device�Your program is NOT input-bound because only 2.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�40.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�27.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 