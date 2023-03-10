�	�k&�lC@�k&�lC@!�k&�lC@	��mА�!@��mА�!@!��mА�!@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�k&�lC@�
a5�0�?1Ae���+@A����t�?I�g���
@Y��S���?rEagerKernelExecute 0*	�G�zLt@2U
Iterator::Model::ParallelMapV2Φ#����?!����@�G@)Φ#����?1����@�G@:Preprocessing2F
Iterator::Model��\���?!.^�b�3P@)(��h���?1�&6S�>1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatz��{�?!L@��>,@)2��У?1�g��~�'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice&4I,)w�?!���l~�"@)&4I,)w�?1���l~�"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�}��A�?!�C�:c�A@)g����?1[+8S@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�u�?!	��;*+@)�����?1�@6�W-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor<2V��W}?!nuR�ӥ@)<2V��W}?1nuR�ӥ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�q�@H�?!����,@)���<j?1�:X�+]�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 8.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�43.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t16.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9��mА�!@I��֍�BN@Q�,|��>@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�
a5�0�?�
a5�0�?!�
a5�0�?      ��!       "	Ae���+@Ae���+@!Ae���+@*      ��!       2	����t�?����t�?!����t�?:	�g���
@�g���
@!�g���
@B      ��!       J	��S���?��S���?!��S���?R      ��!       Z	��S���?��S���?!��S���?b      ��!       JGPUY��mА�!@b q��֍�BN@y�,|��>@�"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter6�-9t�?!6�-9t�?0"1
model/Conv1D_3/conv1dConv2D�ĕ�%�?!]�aa�H�?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGradI�*%X�?!o������?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInputߊ��F�?!�U��p�?0"C
%gradient_tape/model/Conv1D_2/ReluGradReluGrad���1�ߢ?!�抚���?"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput%���B�?!��6	��?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput^���?!������?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�@L�?!�{<�4�?0"1
model/Conv1D_2/conv1dConv2D�k�62�?!п�_��?"\
=model/Conv1D_2/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose��5Ť�?!��>����?Q      Y@Ym۶m۶)@a�$I�$�U@qdi?�Y�A@y�;�>��?"�
both�Your program is MODERATELY input-bound because 8.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�43.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t16.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�35.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 