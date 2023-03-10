�	�M� @@�M� @@!�M� @@	�B>��@�B>��@!�B>��@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0�M� @@$c���@1��B=!0@I�`��'$@YE�J��?r0*	���Q b@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��	���?!z��1>A@)��аu�?1�.Ð��<@:Preprocessing2U
Iterator::Model::ParallelMapV2��,&�?!D0j�M4@)��,&�?1D0j�M4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap@i�QH2�?!�/�O��9@)�fc%�Y�?1����1@:Preprocessing2F
Iterator::ModelMi�-��?!�̈�B$@@)�v/�ɑ?1��©��'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceR���T�?!�i��А!@)R���T�?1�i��А!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���_#�?!��;���P@)��ؙB�?1@f�\�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��
a5��?!Q���,W@)��
a5��?1Q���,W@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 14.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�31.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�B>��@I�$��v�F@Q��@� I@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	$c���@$c���@!$c���@      ��!       "	��B=!0@��B=!0@!��B=!0@*      ��!       2      ��!       :	�`��'$@�`��'$@!�`��'$@B      ��!       J	E�J��?E�J��?!E�J��?R      ��!       Z	E�J��?E�J��?!E�J��?b      ��!       JGPUY�B>��@b q�$��v�F@y��@� I@�".
IteratorGetNext/_29_Send{z�!��?!{z�!��?".
IteratorGetNext/_31_Send#\K�n�?!��x�v�?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInputc���oR�?!Uq�r�k�?0"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/SparseDotIncBuilder/transpose_grad/transpose	Transposed.w� ��?!;�!{���?"�
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolutionConv2D\��X���?!��� :s�?"�
{keras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/SparseDotIncBuilder/SparseTensorDenseMatMulSparseTensorDenseMatMul1P��^�?!�!��#��?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/SimNeuronsBuilder/Relu_grad/ReluGradReluGrad;pAGNő?!\�B�j�?"K
$mean_squared_error/SquaredDifferenceSquaredDifferencez��#"_�?!��7S���?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/AddN_19AddNG#��=�?!����?"=
$gradient_tape/mean_squared_error/subSub�>(�}��?!�D�Ѱ�?Q      Y@Y������(@a������U@q��wP���?y��䓏�?"�

both�Your program is POTENTIALLY input-bound because 14.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�31.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 