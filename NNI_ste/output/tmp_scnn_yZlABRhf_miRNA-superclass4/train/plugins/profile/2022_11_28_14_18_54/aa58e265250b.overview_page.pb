�	�����^@�����^@!�����^@	k�V���%@k�V���%@!k�V���%@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0�����^@��?�?11zn��H@IzT��M@Y`��s�*@r0*	W9��6��@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice �!p�!@!f#�uׁX@) �!p�!@1f#�uׁX@:Preprocessing2F
Iterator::Modelhx�﫲?!�ɳim��?)�����?1�A�W���?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat%�c\qq�?!�����?)��f���?1�c�JF�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapt��z��!@!	G��C�X@)ڨN���?1��#B{l�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�\4d<�!@!m�,%O�X@)�����?1K�5U*��?:Preprocessing2U
Iterator::Model::ParallelMapV2�׻?ޫ�?!v>H�a�?)�׻?ޫ�?1v>H�a�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor2Ƈ�˶�?!KV-�I�?)2Ƈ�˶�?1KV-�I�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 11.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�48.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9k�V���%@I��?d�RH@Q�y*��2D@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��?�?��?�?!��?�?      ��!       "	1zn��H@1zn��H@!1zn��H@*      ��!       2      ��!       :	zT��M@zT��M@!zT��M@B      ��!       J	`��s�*@`��s�*@!`��s�*@R      ��!       Z	`��s�*@`��s�*@!`��s�*@b      ��!       JGPUYk�V���%@b q��?d�RH@y�y*��2D@�".
IteratorGetNext/_27_SendP���%��?!P���%��?".
IteratorGetNext/_25_Sendۆ� +b�?!ߞ퇝0�?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_4/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter��}�?!q{�)M��?0".
IteratorGetNext/_29_SendK
��̢?!���!�J�?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInputJ�^9��?!8��C��?0"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter'_8�|�?!�8�|�?0"�
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_4/convolutionConv2D�ՑJ���?!o��.��?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_4/convolution_grad/Conv2DBackpropInputConv2DBackpropInputR@�*M�?!H�j����?0".
IteratorGetNext/_31_Send���d��?!D��b�?"�
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_3/convolutionConv2D��WB�ȓ?!P�� �?Q      Y@YsNc~,'@a2?�3pV@q>f��"@�?yL���Ww�?"�

both�Your program is MODERATELY input-bound because 11.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�48.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 