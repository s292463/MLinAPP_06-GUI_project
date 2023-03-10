�	�7��@�7��@!�7��@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�7��@ ������?1A�ȓ��@Aly�z�L�?I~�ƃ-�@rEagerKernelExecute 0*	������d@2F
Iterator::ModelqǛ���?!2L*F�H@)�
�|$%�?1.w��@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��`�d�?!(�K A;@)�?�Ŋ�?1(ᳬ�k7@:Preprocessing2U
Iterator::Model::ParallelMapV2>�*�?!�v��&,@)>�*�?1�v��&,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��J�?!f�i���(@)��J�?1f�i���(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateI/j�� �?!��_2@)�&OYM׃?1��e@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	ސFN�?!ͳչ.�I@)�ɋL���?1��<֚@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor.v��2Sz?!�I�,�@).v��2Sz?1�I�,�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap	�^)ˠ?!A�Ƙ��3@)D6�.6�d?1�
kv��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 7.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�52.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��ܓJ�M@Q$#l�'D@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	 ������? ������?! ������?      ��!       "	A�ȓ��@A�ȓ��@!A�ȓ��@*      ��!       2	ly�z�L�?ly�z�L�?!ly�z�L�?:	~�ƃ-�@~�ƃ-�@!~�ƃ-�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��ܓJ�M@y$#l�'D@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��ε�D�?!��ε�D�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��O���?!$#���%�?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad���b��?!x��n���?"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad��!�?!h��j��?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad[�;1$�?!��w��?"1
model/Conv1D_2/conv1dConv2D��47��?!7�"��?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transposez<w���?!�]6��?"{
\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizer	Transposes�F*�r�?!f���7��?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�i֔� �?!�0(�B��?0"}
^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose,�j�1�?!�ۮ����?Q      Y@Y     �-@a     JU@qv��#E@y���5㪱?"�
both�Your program is POTENTIALLY input-bound because 7.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�52.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�42.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 