�	?���;@?���;@!?���;@      ��!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?���;@1PoF�w@I8I�Ǵz5@r0*	������s@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap!��	L��?!ı&!�L@)��"ڎ��?1�S{J@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat+j0�?!�@���0@)�gs��?1��U%�+@:Preprocessing2F
Iterator::Model=�Ƃ �?!x��Ok1@)9DܜJ�?1��-b��#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���i��?!�,%�T@)�ʼUס�?13ƴ�#G @:Preprocessing2U
Iterator::Model::ParallelMapV2V���4�?!��z6�@)V���4�?1��z6�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�w��ۏ?!�-F�x@)�w��ۏ?1�-F�x@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�w�*�?!��eo5@)�w�*�?1��eo5@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�77.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI.7��HS@QF#�a�6@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
      ��!             ��!       "	PoF�w@PoF�w@!PoF�w@*      ��!       2      ��!       :	8I�Ǵz5@8I�Ǵz5@!8I�Ǵz5@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q.7��HS@yF#�a�6@�".
IteratorGetNext/_26_Recv������?!������?".
IteratorGetNext/_28_Recv�8'�t�?!&#%49�?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_4/convolution_grad/Conv2DBackpropInputConv2DBackpropInputgLI�ˤ?! ��D�5�?0"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_4/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterU�HF��?!%�\�q��?0".
IteratorGetNext/_30_Recv0ӗ�	�?!H��Br�?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_3/convolution_grad/Conv2DBackpropInputConv2DBackpropInputn}�Oy��?!��ג��?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamƴxq��?!k����?"�
lkeras_model/TensorGraph/while/body/_1/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_4/convolutionConv2D=y�H��?!�l���?"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/ConvIncBuilder_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterU��mM�?!�ݔ��0�?0"�
�gradient_tape/keras_model/TensorGraph/while/keras_model/TensorGraph/while_grad/body/_631/gradient_tape/keras_model/TensorGraph/while/gradients/keras_model/TensorGraph/while/iteration_0/SparseDotIncBuilder/SparseTensorDenseMatMul_grad/SparseTensorDenseMatMulSparseTensorDenseMatMulh�}Y�7�?!w�`p~"�?Q      Y@Yo��o��'@a� � V@q�@4RS@yf4"���?"�

device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�77.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 