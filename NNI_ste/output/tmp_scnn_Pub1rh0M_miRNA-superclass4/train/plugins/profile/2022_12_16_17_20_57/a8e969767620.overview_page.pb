�	6�EaxT@6�EaxT@!6�EaxT@	���o��"@���o��"@!���o��"@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails06�EaxT@��'�.��?1 ��q�yN@I3�z�w)@Y:"ߥ��@r0*	���Se@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�]gC���?!%]w9\�=@)���4)�?1PfaO�K8@:Preprocessing2U
Iterator::Model::ParallelMapV2"�4��?!؊ >��2@)"�4��?1؊ >��2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��2p@�?!��U��/@)��2p@�?1��U��/@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�K�K�1�?!ާ��F>@)���d#�?1Z����-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�gB�Ē�?!���=�Q@)�J�.���?1.Vi�%@:Preprocessing2F
Iterator::Model��6�^�?!���S=@)������?1�W;�%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���:TS�?!V�W�k.@)���:TS�?1V�W�k.@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�15.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���o��"@I,�=��$0@Q�68�R@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��'�.��?��'�.��?!��'�.��?      ��!       "	 ��q�yN@ ��q�yN@! ��q�yN@*      ��!       2      ��!       :	3�z�w)@3�z�w)@!3�z�w)@B      ��!       J	:"ߥ��@:"ߥ��@!:"ߥ��@R      ��!       Z	:"ߥ��@:"ߥ��@!:"ߥ��@b      ��!       JGPUY���o��"@b q,�=��$0@y�68�R@�".
IteratorGetNext/_29_Send�I�֢�?!�I�֢�?".
IteratorGetNext/_31_Send�.�F�?!��%8��?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput�ŘUc�?!1���R2�?0"�
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_1/convolutionConv2D96Fd1!�?!��]\��?".
IteratorGetNext/_25_Send��[�L}�?!�h��Q%�?"�
{keras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/SparseDotIncBuilder/SparseTensorDenseMatMulSparseTensorDenseMatMul�[>�9�?!b>wޔ�?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/SimNeuronsBuilder/Relu_grad/ReluGradReluGrad� \R|>�?!"҇h���?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/SparseDotIncBuilder/transpose_grad/transpose	Transpose����+�?!��Ҍ�f�?"K
$mean_squared_error/SquaredDifferenceSquaredDifference�tWBb�?!1�0���?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_611/gradient_tape/keras_model/TensorGraph/while/gradients/AddN_19AddNOn�q+�?!��ϻ(�?Q      Y@Y$?��5�(@ax�N��U@q
[��e�?y�����u?"�

both�Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�15.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 