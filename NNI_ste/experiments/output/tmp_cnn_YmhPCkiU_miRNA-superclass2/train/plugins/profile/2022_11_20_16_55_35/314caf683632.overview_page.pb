�	0�k���A@0�k���A@!0�k���A@	V3w�	:�?V3w�	:�?!V3w�	:�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL0�k���A@�b9�?1�0|D�@Ar�Md��?I>"�D�=@Y�aN�&�?rEagerKernelExecute 0*	��"��fd@2F
Iterator::Model�n��?!��E	�FC@)��V���?1]��a�8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�d8�π�?!���<$6@)�d8�π�?1���<$6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��iܣ?!!��1�7@)]��X32�?1�*5�}a3@:Preprocessing2U
Iterator::Model::ParallelMapV2��OV�?!������+@)��OV�?1������+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�����?!'0��+�N@)(`;�O�?1�nj?Ȅ#@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap:�6U�?!�eں��;@)�|ԛQ�?1��a�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�y��Q}?!a�~̊@)�y��Q}?1a�~̊@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�82.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9U3w�	:�?I%����sU@Qmݯ�Y*@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�b9�?�b9�?!�b9�?      ��!       "	�0|D�@�0|D�@!�0|D�@*      ��!       2	r�Md��?r�Md��?!r�Md��?:	>"�D�=@>"�D�=@!>"�D�=@B      ��!       J	�aN�&�?�aN�&�?!�aN�&�?R      ��!       Z	�aN�&�?�aN�&�?!�aN�&�?b      ��!       JGPUYU3w�	:�?b q%����sU@ymݯ�Y*@�"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter-��~�Y�?!-��~�Y�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��S��?!
(��$��?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�S>��%�?!�Qn'��?0"1
model/Conv1D_2/conv1dConv2D�2,tJa�?!�5�0�5�?"1
model/Conv1D_3/conv1dConv2DT�Qy{�?!���ZBe�?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad����8�?!�s�Tl�?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput!N�󖾧?!�{�'d�?0"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad����}�?!�Q!0�s�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGradAl E��?!Z�A6�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter����L��?!.L��@��?0Q      Y@Ywb'vb'*@a�;��U@qhɚ�S@@y��?!?a�?"�
device�Your program is NOT input-bound because only 1.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�82.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�32.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 