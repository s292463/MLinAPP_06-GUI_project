�	��M��@��M��@!��M��@	2Cn_8�@2Cn_8�@!2Cn_8�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��M��@��.���?1���mW�?Aut\��J�?I�Gp#e+@Y"¿3�?rEagerKernelExecute 0*	
ףp=~d@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���j�?!��8�A�=@)Xr�ߤ?1_+Y���8@:Preprocessing2F
Iterator::Model���-��?!h���hB@)��Ϸ�?1$��3p�6@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceA�M�Gş?!WG*��2@)A�M�Gş?1WG*��2@:Preprocessing2U
Iterator::Model::ParallelMapV2N�»\ė?!%���zP,@)N�»\ė?1%���zP,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip+Kt�Y��?!�=3)�O@)���[��?1�nv��9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�2nj���?!�I?~��8@)�o�[�?1��O%A@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor=�r�}ǀ?!`�}oX�@)=�r�}ǀ?1`�}oX�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap͓k
dv�?!o�d��:@)���9�g?1[B�Y�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�43.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t24.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.93Cn_8�@IMnZ� Q@Q���r�9@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��.���?��.���?!��.���?      ��!       "	���mW�?���mW�?!���mW�?*      ��!       2	ut\��J�?ut\��J�?!ut\��J�?:	�Gp#e+@�Gp#e+@!�Gp#e+@B      ��!       J	"¿3�?"¿3�?!"¿3�?R      ��!       Z	"¿3�?"¿3�?!"¿3�?b      ��!       JGPUY3Cn_8�@b qMnZ� Q@y���r�9@�"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilterZ�w�l�?!Z�w�l�?0"1
model/Conv1D_3/conv1dConv2D��m�&�?!��rN��?"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�E��Ǯ?!иG{�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput*8�1�?!FW����?0"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad��~"�?!����0�?"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput|��n�?!?y�1��?0"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits��[gB��?!2��~���?"1
model/Conv1D_2/conv1dConv2D�,@T�?!b$���?"W
6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradMaxPoolGrad�e�7(S�?!�����?"C
%gradient_tape/model/Conv1D_2/ReluGradReluGradF^��n�?!w���m��?Q      Y@Y�ܺ�+@a�p�h�U@q`�K�tI:@y����Vb�?"�
both�Your program is MODERATELY input-bound because 5.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�43.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t24.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�26.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 