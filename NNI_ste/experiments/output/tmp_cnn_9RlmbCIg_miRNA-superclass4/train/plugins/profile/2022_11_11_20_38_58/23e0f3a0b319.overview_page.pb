�	��@�g@��@�g@!��@�g@	�Fa�?�Fa�?!�Fa�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��@�g@�~P)��?1����c@A�3M�~2�?I!��=@�=@Yv5y�j��?rEagerKernelExecute 0*	����Mba@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateiUMu�?!���C�x@@)�=yX��?1�غfxj>@:Preprocessing2F
Iterator::Modelb�[>���?!C	��wE@)���1v£?1>���;@:Preprocessing2U
Iterator::Model::ParallelMapV2`L8��?!�B�_.@)`L8��?1�B�_.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�M���P�?!���0�L@)����q�?1�|ߣE#@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatf/�N[#�?!�y�ty)@)�bE�a�?1��|�!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��� !�w?!O���{�@)��� !�w?1O���{�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�\S ���?!��;XA@)Z��/-�c?1�j���?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor`x%�s}_?!ّ�پ�?)`x%�s}_?1ّ�پ�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�ù�Z?!F��5�S�?)�ù�Z?1F��5�S�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�16.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�Fa�?I��_��0@QS!�'��T@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�~P)��?�~P)��?!�~P)��?      ��!       "	����c@����c@!����c@*      ��!       2	�3M�~2�?�3M�~2�?!�3M�~2�?:	!��=@�=@!��=@�=@!!��=@�=@B      ��!       J	v5y�j��?v5y�j��?!v5y�j��?R      ��!       Z	v5y�j��?v5y�j��?!v5y�j��?b      ��!       JGPUY�Fa�?b q��_��0@yS!�'��T@�"1
model/Conv1D_3/conv1dConv2D��lm�?!��lm�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterE��ٜ��?!d�n��?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput#����?!{��8��?0"1
model/Conv1D_4/conv1dConv2D^����@�?!GH���?"1
model/Conv1D_2/conv1dConv2D/�ZL�?!)�v͒�?"1
model/Conv1D_1/conv1dConv2D4�nU���?!��u~�?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter������?!t^P�=X�?0"b
7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputConv2DBackpropInput
��vnL�?!Uh���?0"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter܎D���?!C������?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputl��ۇ�?!Z]�7�.�?0Q      Y@Y�Cc}h@aO��)x�W@q��@�d@@yJ�%��h?"�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�16.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�32.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 