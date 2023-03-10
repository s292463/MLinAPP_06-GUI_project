�	�#~��(@�#~��(@!�#~��(@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�#~��(@������?1�E��@AƦ�B ��?I��"�n@rEagerKernelExecute 0*	�z�GIc@2F
Iterator::Model�ơ~��?!sp��G@)����]i�?1Q��ހ@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateXo�
���?!"��,�A@)��ډ���?1�`���?@:Preprocessing2U
Iterator::Model::ParallelMapV2�/ע�?!��D�h.@)�/ע�?1��D�h.@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatQ�+�ϒ?!�u�I�'@)�Z��8�?1i��e@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��w)uɴ?!����RPJ@)��P�l}?1�*;�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor73��p�|?!��39@)73��p�|?1��39@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��=z�}�?!����B@)tF��_h?1 ������?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�� ��ze?!oQ�(�0�?)�� ��ze?1oQ�(�0�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorV��L�`?!�JƝ�|�?)V��L�`?1�JƝ�|�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 15.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�22.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noID�c��C@Q�I�V�N@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	������?������?!������?      ��!       "	�E��@�E��@!�E��@*      ��!       2	Ʀ�B ��?Ʀ�B ��?!Ʀ�B ��?:	��"�n@��"�n@!��"�n@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qD�c��C@y�I�V�N@�"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter���|��?!���|��?0"1
model/Conv1D_2/conv1dConv2DO��e���?!�㗓��?"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInputm��{���?!(9�v���?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrado�&`�?!ϔ4y���?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad�����Y�?!TnR~��?"d
8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�,yFk�?!���2s�?0"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter2���e=�?!`�Ջ��?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad�����?!^8����?"1
model/Conv1D_3/conv1dConv2D�_�'E�?!E1����?"\
=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizer	Transpose�>�?!.�֝�N�?Q      Y@YI�$I�$+@a�m۶m�U@qM��a=M@y��{Ŧ�?"�
both�Your program is POTENTIALLY input-bound because 15.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�22.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�58.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 