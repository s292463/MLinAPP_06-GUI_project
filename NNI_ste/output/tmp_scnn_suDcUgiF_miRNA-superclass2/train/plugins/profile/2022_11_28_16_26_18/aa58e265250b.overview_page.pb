�	���vh�E@���vh�E@!���vh�E@	� ��6@� ��6@!� ��6@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0���vh�E@��ފ��?1��vN��=@Ii�hs��$@Y
pU*
@r0*�K7�A�l@)       =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat7������?!�@�9@)m��~���?1��2H25@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�)t^c��?!��}.�l<@)���1�3�?1����3@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���`��?!�k�X�Q@)�zO崧�?1�*�x�1@:Preprocessing2U
Iterator::Model::ParallelMapV2�-u�׃�?!�Dsc�.@)�-u�׃�?1�Dsc�.@:Preprocessing2F
Iterator::Model?�ܵ�|�?!ZP֜�><@)����c�?1&\9�~{*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceE� ���?!�H�8!@)E� ���?1�H�8!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor&�v��-�?!��GK7$@)&�v��-�?1��GK7$@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�23.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9� ��6@I���9�7@Q`Gv�Q@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��ފ��?��ފ��?!��ފ��?      ��!       "	��vN��=@��vN��=@!��vN��=@*      ��!       2      ��!       :	i�hs��$@i�hs��$@!i�hs��$@B      ��!       J	
pU*
@
pU*
@!
pU*
@R      ��!       Z	
pU*
@
pU*
@!
pU*
@b      ��!       JGPUY� ��6@b q���9�7@y`Gv�Q@�".
IteratorGetNext/_25_Send����4��?!����4��?".
IteratorGetNext/_29_SendW*
�9�?!`E�z���?".
IteratorGetNext/_27_Send�Q0:�:�?!�Yp	6��?".
IteratorGetNext/_31_Send��_���?!RW� �y�?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_6/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter|��4'~�?!j�)tl!�?0"�
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_6/convolutionConv2D�A�끞�?!�&�T;�?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter�aA��?!�<�M�?0"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput�Y١�D�?!DҺ�SQ�?0"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_5/convolution_grad/Conv2DBackpropInputConv2DBackpropInput���덉?!騂U��?0"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/SparseDotIncBuilder/transpose_grad/transpose	Transpose
�}D⚈?!q��g���?Q      Y@Y��Id�'@aL�v�<	V@q�Re��?y���y
C|?"�

both�Your program is MODERATELY input-bound because 7.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�23.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 