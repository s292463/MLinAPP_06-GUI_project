�	TS�u8:!@TS�u8:!@!TS�u8:!@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCTS�u8:!@HQg�!��?19�Z���@A����ׁ�?I�hr1�@rEagerKernelExecute 0*	�C�l�c@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZ�>�-W�?!�D)��=@)��+I��?1ܩ:w~9@:Preprocessing2F
Iterator::ModelQS�'��?!��H/7�B@)�L�n�?1���/To8@:Preprocessing2U
Iterator::Model::ParallelMapV2��+�,�?!�"�]4�*@)��+�,�?1�"�]4�*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��d#ٓ?!1�O�ff)@)��d#ٓ?11�O�ff)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��M(D�?!n1���O@)�4��?1^��V�&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��H�,|�?!�N�~��2@)���	F�?1L��&4�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor겘�|\{?!i�%�ށ@)겘�|\{?1i�%�ށ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapܞ ��=�?!7Lg��4@)��v�g?1��?���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 11.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�54.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�ۢBO�P@QH�za�@@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	HQg�!��?HQg�!��?!HQg�!��?      ��!       "	9�Z���@9�Z���@!9�Z���@*      ��!       2	����ׁ�?����ׁ�?!����ׁ�?:	�hr1�@�hr1�@!�hr1�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�ۢBO�P@yH�za�@@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�ȡt̆�?!�ȡt̆�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�Z���?!䑫>�K�?0"1
model/Conv1D_2/conv1dConv2D|B�(�?!"�1�_�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad}盛?!��;xP��?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad[X=׎��?!��#SBy�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�N4�С?!���X��?0"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transposeb�	K�?!�`����?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�ƛ�H�?!d��8��?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�'����?!b�LS(�?"3
model/Conv1D_1/BiasAddBiasAdd���s�h�?!�G���?Q      Y@Y     �'@a     V@q�����@B@yo���?"�
both�Your program is POTENTIALLY input-bound because 11.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�54.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�36.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 