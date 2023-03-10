�	�����@�����@!�����@	55��qF@55��qF@!55��qF@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�����@�	h"lx�?1IIC��?AX:�%Ȩ?I�F��@Y� \�z�?rEagerKernelExecute 0*	/�$�u@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateRcB�%U�?!#���'[J@)�p�Qe�?1T>��/�G@:Preprocessing2F
Iterator::Model9��ㄵ?!�1�H�N8@)�_�L�?1���*2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���Y��?!Ra�H
0@)&��s|��?1[�tQ�+@:Preprocessing2U
Iterator::Model::ParallelMapV2G�@�]��?!ٲ悩�@)G�@�]��?1ٲ悩�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipq $���?!�sέP�R@)8�0��?1�)�BU�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice:�w��?!���7@):�w��?1���7@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�&p�n~?!'���<0@)�&p�n~?1'���<0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�4�ׂ��?!Eq�Q�J@)1zn�+q?1Jd��8e�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�50.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t23.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.955��qF@I-@p�s�R@Q�����4@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�	h"lx�?�	h"lx�?!�	h"lx�?      ��!       "	IIC��?IIC��?!IIC��?*      ��!       2	X:�%Ȩ?X:�%Ȩ?!X:�%Ȩ?:	�F��@�F��@!�F��@B      ��!       J	� \�z�?� \�z�?!� \�z�?R      ��!       Z	� \�z�?� \�z�?!� \�z�?b      ��!       JGPUY55��qF@b q-@p�s�R@y�����4@�"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�/J��J�?!�/J��J�?0"1
model/Conv1D_3/conv1dConv2D_>��ϱ?!7�mN�?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput��,J��?!������?0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad۩�t�?!��=Ö��?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput>־���?!����i�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��Px�Y�?!��"ʿ��?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad��f�h��?!o�/�k�?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputp���XO�?!}A����?0"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrady{$��1�?!5g#�h�?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter3���?�?!ءlX��?0Q      Y@Ym۶m۶)@a�$I�$�U@qw���l�6@y��e�]��?"�
both�Your program is MODERATELY input-bound because 5.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�50.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t23.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�22.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 