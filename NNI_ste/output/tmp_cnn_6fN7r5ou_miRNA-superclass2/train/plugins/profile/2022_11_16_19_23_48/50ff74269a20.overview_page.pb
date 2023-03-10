�	:�%�8@:�%�8@!:�%�8@	����@����@!����@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL:�%�8@�'eRC�?1+4�fn@A��}�Az�?I	pz�@Y�Y��!��?rEagerKernelExecute 0*	l����c@2F
Iterator::Model	��g��?!�0�$�H@)jj�Z_$�?1_oR�{wA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatB�۽�'�?!���f�<@)8>[�?1�o'��7@:Preprocessing2U
Iterator::Model::ParallelMapV2R��b�?!@�x;�-@)R��b�?1@�x;�-@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicep@KW���?!�_㒔z@)p@KW���?1�_㒔z@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�|��⋖?!��%���+@)Z����?1�AhW�~@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip ��P�\�?!i�k�FI@)T����#�?1���;|�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(v�U��?!��cv~@)(v�U��?1��cv~@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapRD�U���?!Q�Kb�/@)i;���.h?1��.���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 16.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�31.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9����@I��E��ZH@QA�F���G@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�'eRC�?�'eRC�?!�'eRC�?      ��!       "	+4�fn@+4�fn@!+4�fn@*      ��!       2	��}�Az�?��}�Az�?!��}�Az�?:		pz�@	pz�@!	pz�@B      ��!       J	�Y��!��?�Y��!��?!�Y��!��?R      ��!       Z	�Y��!��?�Y��!��?!�Y��!��?b      ��!       JGPUY����@b q��E��ZH@yA�F���G@�"1
model/Conv1D_2/conv1dConv2D���
�
�?!���
�
�?"d
8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��i��k�?!kd�cc;�?0"d
8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilterx���?!'�j�+�?0"b
7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputConv2DBackpropInput���D�?!�Ofd��?0"b
7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputConv2DBackpropInput����Ψ?!�":��?0"1
model/Conv1D_3/conv1dConv2D,j�����?!��
)��?"W
6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradMaxPoolGrad=(H�[��?!C@�~{�?"d
8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter��G.�`�?!��c��1�?0"W
6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradMaxPoolGrad$��%G��?!�� ��?"C
%gradient_tape/model/Conv1D_1/ReluGradReluGrad��b6��?!4#��Z�?Q      Y@Y      )@a     �U@q���0��6@yyT=���?"�
both�Your program is POTENTIALLY input-bound because 16.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�31.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�22.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 