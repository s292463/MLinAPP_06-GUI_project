�	���G� @���G� @!���G� @	v�ޑ��@v�ޑ��@!v�ޑ��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC���G� @�'�XQ�?1�F� �@I�e3��v@Y���n��?rEagerKernelExecute 0*	ˡE��g@2F
Iterator::Modelj1x��͵?!`��\�oF@)X�2ı.�?1rc�H�?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���I��?!&z�kI8@)_B�D�?1�>�v�4@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��}ɺ?!�p5�?�K@)t#,*�t�?1�W v��2@:Preprocessing2U
Iterator::Model::ParallelMapV2�0{�vښ?!�v�ᶡ+@)�0{�vښ?1�v�ᶡ+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicev��$�p�?!�u��@)v��$�p�?1�u��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�%�"�d�?!഑�.�$@)� �X�?1��5��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�u�ݑ�z?!���v�w@)�u�ݑ�z?1���v�w@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���Y��?!�k�'@)P�<�e?1fO����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�27.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9w�ޑ��@Ih�zB��>@QZR�*exP@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�'�XQ�?�'�XQ�?!�'�XQ�?      ��!       "	�F� �@�F� �@!�F� �@*      ��!       2      ��!       :	�e3��v@�e3��v@!�e3��v@B      ��!       J	���n��?���n��?!���n��?R      ��!       Z	���n��?���n��?!���n��?b      ��!       JGPUYw�ޑ��@b qh�zB��>@yZR�*exP@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilteri�0wd߯?!i�0wd߯?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter\۹g釦?!�8u�3�?0"C
%gradient_tape/model/Conv1D_1/ReluGradReluGradE{d�e�?!B��,�?"1
model/Conv1D_2/conv1dConv2D�3<~#��?!9Ȣȵ=�?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad%��>�q�?!��EXZ�?"1
model/Conv1D_3/conv1dConv2D���X�£?!`
<^%�?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transpose�n�ŀ�?!|�)�v5�?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	TransposeB����?!d�a�wE�?"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose ٟ��;�?!��u;�L�?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter8X"���?!�"�-gO�?0Q      Y@Y��k(�'@a����
V@q����2@y[^��ʦ?"�
both�Your program is POTENTIALLY input-bound because 3.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�27.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�18.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 