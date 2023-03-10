�	�w�7N�>@�w�7N�>@!�w�7N�>@	+��r�@+��r�@!+��r�@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0�w�7N�>@�i�����?1N��ol/@I�VC�V,@Ym��}��?r0*	-���r�@2Z
#Iterator::Model::ParallelMapV2::ZipP��n�?!誱�1�V@)�/�$�?1�[Ѷ��R@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat\�v5y�?!��c=�!@)��4��?1��`ǭ�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�m��ʆ�?!�ڂ��|@).���=��?1����<@:Preprocessing2F
Iterator::Model���qn�?!��r"q�#@)�@1�d�?1�hp=6@:Preprocessing2U
Iterator::Model::ParallelMapV21[�*?!��t��@)1[�*?1��t��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice6\䞮�?!f��2 @)6\䞮�?1f��2 @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J=By?!��� 4��?)�J=By?1��� 4��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 2.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�46.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9+��r�@I�8NZ1G@Q9K����I@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�i�����?�i�����?!�i�����?      ��!       "	N��ol/@N��ol/@!N��ol/@*      ��!       2      ��!       :	�VC�V,@�VC�V,@!�VC�V,@B      ��!       J	m��}��?m��}��?!m��}��?R      ��!       Z	m��}��?m��}��?!m��}��?b      ��!       JGPUY+��r�@b q�8NZ1G@y9K����I@�".
IteratorGetNext/_30_Recv=��Xk�?!=��Xk�?".
IteratorGetNext/_31_Send���Xð?!;�/���?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput��(�3�?!VԹ�W��?0"�
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolutionConv2Djz�o:��?!�볓��?"�
{keras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/SparseDotIncBuilder/SparseTensorDenseMatMulSparseTensorDenseMatMul��鑐�?!��ҜF�?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/SparseDotIncBuilder/transpose_grad/transpose	Transpose���;�_�?!��GX 6�?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/SimNeuronsBuilder/Relu_grad/ReluGradReluGrad5�J=ϰ�?!>I2҆��?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/AddN_19AddN˖��C�?!�_�L�?"K
$mean_squared_error/SquaredDifferenceSquaredDifference�0��e�?!{���?"}
bkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/CopyBuilder_10/addAddV2g�s���?!.EcQY�?Q      Y@Y$?��5�(@ax�N��U@q/�3 @y1O^��
�?"�

device�Your program is NOT input-bound because only 2.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�46.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 