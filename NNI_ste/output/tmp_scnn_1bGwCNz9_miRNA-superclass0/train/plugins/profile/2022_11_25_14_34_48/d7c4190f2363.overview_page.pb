�	����=@����=@!����=@	X�5]YR@X�5]YR@!X�5]YR@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0����=@���*P��?1����0@I?Ȳ`�w&@Y�@J��?r0*	��ʡ��@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map��:8�#@!�>��;V@)��D�"@1�{95�V@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�k	��'�?!��q��$@)��3K��?1C�9��#@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat��XP��?!8Dì���?)�&�����?1�F�{Q��?:Preprocessing2F
Iterator::Model��L����?!�K�����?)i���?18�����?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���Wy�?!n�0�ǐ�?)�;��J"�?1������?:Preprocessing2U
Iterator::Model::ParallelMapV2����w��?!H��rW��?)����w��?1H��rW��?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipI����?!�3�,�%@)2<��X��?1�}BP��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorkׄ�Ơ�?!fB���?)kׄ�Ơ�?1fB���?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice2 Tq��?!{&at�H�?)2 Tq��?1{&at�H�?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetchy\T��b�?!o^(v�?)y\T��b�?1o^(v�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangeܸ����t?!��߈�L�?)ܸ����t?1��߈�L�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Wa?!4T���>�?)����Wa?14T���>�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�37.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Y�5]YR@I|X��C@Q������K@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���*P��?���*P��?!���*P��?      ��!       "	����0@����0@!����0@*      ��!       2      ��!       :	?Ȳ`�w&@?Ȳ`�w&@!?Ȳ`�w&@B      ��!       J	�@J��?�@J��?!�@J��?R      ��!       Z	�@J��?�@J��?!�@J��?b      ��!       JGPUYY�5]YR@b q|X��C@y������K@�".
IteratorGetNext/_29_Sendm?�E�?!m?�E�?".
IteratorGetNext/_31_Send�L�H��?!�����?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterU�k�3�?!�2x}��?0"�
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolutionConv2DA�B�@��?!���9�?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInputD����*�?!���Ne~�?0"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInputڛh���?!�t��@�?0"�
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolutionConv2D�U�@1g�?!g�?���?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput��c�Q�?!S�����?0"�
{keras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/SparseDotIncBuilder/SparseTensorDenseMatMulSparseTensorDenseMatMul�2|��?!rZ��?"�
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_3/convolutionConv2D4�{�ߋ?!����
��?Q      Y@Y���M��,@a�`H��fU@q�rUH���?y�"�Mi�?"�

both�Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�37.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 