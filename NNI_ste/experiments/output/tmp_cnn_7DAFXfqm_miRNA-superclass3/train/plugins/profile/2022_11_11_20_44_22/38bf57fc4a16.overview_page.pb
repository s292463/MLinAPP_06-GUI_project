�	�H�"�@�H�"�@!�H�"�@	B'z%J@B'z%J@!B'z%J@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�H�"�@��_ѭ�?1�BB�@A$C��g�?I3��bb@Y�{*�=%�?rEagerKernelExecute 0*	�����c@2F
Iterator::Model�*���ڳ?!-,]�KH@)�bG�P��?1IW=���@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��F;n��?!a�	���:@)���i��?1�Y$3�6@:Preprocessing2U
Iterator::Model::ParallelMapV2��,��?!�S� E-@)��,��?1�S� E-@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceP�R)v�?!اFM	)@)P�R)v�?1اFM	)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��Wy�?!�Ӣ=��I@)�<��?1:tm}��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�wE𿕜?!�-��|1@)-y<-?�?1�dO�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor0��!�z?!:r�[�N@)0��!�z?1:r�[�N@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%���}��?!7��UN3@)\��b��g?138���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 18.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�37.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9B'z%J@I�dK&]L@Q6�o*��C@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��_ѭ�?��_ѭ�?!��_ѭ�?      ��!       "	�BB�@�BB�@!�BB�@*      ��!       2	$C��g�?$C��g�?!$C��g�?:	3��bb@3��bb@!3��bb@B      ��!       J	�{*�=%�?�{*�=%�?!�{*�=%�?R      ��!       Z	�{*�=%�?�{*�=%�?!�{*�=%�?b      ��!       JGPUYB'z%J@b q�dK&]L@y6�o*��C@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�$�~�?!�$�~�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputs���a5�?!@�f��Y�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradu�m��m�?!�"�1��?"1
model/Conv1D_2/conv1dConv2D\�����?!�g7Dɺ�?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�>�� �?!���z�?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose� Yܮԡ?!�/��_��?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�Z�Q˵�?!����?0"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose\�H#�n�?!�2=���?"3
model/Conv1D_1/BiasAddBiasAdd���p��?!�)V��?"-
model/Conv1D_1/ReluReluMT/��?!#i� �?Q      Y@YI�$I�$+@a�m۶m�U@q,љ`��A@yp7���ش?"�
both�Your program is POTENTIALLY input-bound because 18.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�37.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�35.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 