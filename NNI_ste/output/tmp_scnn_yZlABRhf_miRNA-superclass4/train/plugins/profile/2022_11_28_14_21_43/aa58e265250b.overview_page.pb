�	QMI��4J@QMI��4J@!QMI��4J@      ��!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'QMI��4J@��w�?1H�ξ��A@IN
�g�0@r0*	]�IOj@2F
Iterator::Model���y0�?!9���B@)��3ڪ$�?1(��]�B8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatt]���ԩ?!�j�j��7@)�(�N>�?1T����3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��ʅʟ?!�q�-@)��ʅʟ?1�q�-@:Preprocessing2U
Iterator::Model::ParallelMapV2C��3w�?!��A�Pj*@)C��3w�?1��A�Pj*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��鲘��?!�)�-DO@)j���1>�?1�P��5*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6t�?Pn�?!�䳡�t9@)R,���?1κ���h%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor1AG�Z�?!r��M@)1AG�Z�?1r��M@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�32.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIl�w�X@@QJ0ıS�P@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��w�?��w�?!��w�?      ��!       "	H�ξ��A@H�ξ��A@!H�ξ��A@*      ��!       2      ��!       :	N
�g�0@N
�g�0@!N
�g�0@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb ql�w�X@@yJ0ıS�P@�".
IteratorGetNext/_28_Recv��VG�D�?!��VG�D�?".
IteratorGetNext/_26_RecvvZ��~�?!��a�?".
IteratorGetNext/_30_Recv�7+��֪?!l����?".
IteratorGetNext/_32_Recv8�����?!䉗 `��?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_4/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter��P��?!EȰ.��?0"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_4/convolution_grad/Conv2DBackpropInputConv2DBackpropInputh�9��?!YN��>�?0"�
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_4/convolutionConv2D/�ȝ��?!'�lW�3�?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput)�	)2�?!�e�����?0"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltercS��F��?!S;�?0"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/SparseDotIncBuilder_1/transpose_grad/transpose	Transpose6��~�?!"��*3y�?Q      Y@Y���y;'@aCb�ΐV@q�ڋ �l�?y�tJ���x?"�

device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�32.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 