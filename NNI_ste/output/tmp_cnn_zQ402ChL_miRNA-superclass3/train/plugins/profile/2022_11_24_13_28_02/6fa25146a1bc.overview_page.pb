�	Z��#iy@Z��#iy@!Z��#iy@	�
Ω9g�?�
Ω9g�?!�
Ω9g�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLZ��#iy@U���p�?1{Ùv@A���ٕ?I�5|�E@Y:<��Ӹ�?rEagerKernelExecute 0*	��C�l�b@2F
Iterator::Model]�wb֋�?!��-�@�K@)���ek�?1���K�B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��vö�?!F���v�;@)�I�p�?1�t#�9@:Preprocessing2U
Iterator::Model::ParallelMapV2F�6�X�?!���h�1@)F�6�X�?1���h�1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�u��ݐ?!�8��%@)�2d���?1+�3?+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�"�-�R|?!ߒ=��6@)�"�-�R|?1ߒ=��6@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�i3NCT�?!�1�IF@)d!:�z?1�5�E�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapE�$]3��?!�0z>+�=@)׆�q�&d?1z�8�K��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor�J���>\?!녁�*�?)�J���>\?1녁�*�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�ɐc�Y?!Ա����?)�ɐc�Y?1Ա����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�10.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�
Ω9g�?IȌ�Λ
&@Q�T��?<V@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	U���p�?U���p�?!U���p�?      ��!       "	{Ùv@{Ùv@!{Ùv@*      ��!       2	���ٕ?���ٕ?!���ٕ?:	�5|�E@�5|�E@!�5|�E@B      ��!       J	:<��Ӹ�?:<��Ӹ�?!:<��Ӹ�?R      ��!       Z	:<��Ӹ�?:<��Ӹ�?!:<��Ӹ�?b      ��!       JGPUY�
Ω9g�?b qȌ�Λ
&@y�T��?<V@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�e<�x��?!�e<�x��?0"1
model/Conv1D_2/conv1dConv2D-d�"Qr�?!�d����?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput�xFz
�?!Y�>�!	�?0"1
model/Conv1D_3/conv1dConv2DOsh��]�?!�(uA��?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�����?!ƌ!�s��?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput���cn��?!�JAB���?0"1
model/Conv1D_4/conv1dConv2D��Z�?!���3�?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput���N��?!��N8e��?0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter����M�?!�N���?0"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���#J��?!�%�'rw�?0Q      Y@Y���"�@a�!Tҍ�W@qE���<C@y��r�R?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�38.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 